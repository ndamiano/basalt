// src/core/conversation_file.cpp - the conversation cache's images on disk (see conversation_file.hpp).
#include "basalt/core/conversation_file.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sys/stat.h>

namespace basalt::core {
namespace {

constexpr char kMagic[8] = {'B', 'S', 'L', 'T', 'C', 'O', 'N', 'V'};
constexpr char kTail[8] = {'C', 'O', 'N', 'V', 'E', 'N', 'D', '!'};
constexpr uint32_t kVersion = 1;
constexpr uint64_t kMaxItems = uint64_t(1) << 40;   // any count above this is a damaged file, not a big one

struct FileCloser { void operator()(std::FILE* f) const { if (f) std::fclose(f); } };
using File = std::unique_ptr<std::FILE, FileCloser>;

struct Writer {
    std::FILE* f;
    size_t bytes = 0;
    bool ok = true;
    void raw(const void* p, size_t n) {
        if (ok && n && std::fwrite(p, 1, n, f) != n) ok = false;
        bytes += n;
    }
    template<class T> void pod(const T& v) { raw(&v, sizeof v); }
    void u64(uint64_t v) { pod(v); }
    void str(const std::string& s) { u64(s.size()); raw(s.data(), s.size()); }
    template<class T> void vec(const std::vector<T>& v) { u64(v.size()); raw(v.data(), v.size() * sizeof(T)); }
    void buf(const ConversationBuffer& b) {
        u64(b.size());
        if (!b.visit(0, b.size(), [&](const uint8_t* p, size_t n, size_t) { raw(p, n); return ok; })) ok = false;
    }
};

struct Reader {
    std::FILE* f;
    size_t bytes = 0;
    bool ok = true;
    void raw(void* p, size_t n) {
        if (ok && n && std::fread(p, 1, n, f) != n) ok = false;
        bytes += n;
    }
    template<class T> void pod(T& v) { raw(&v, sizeof v); }
    uint64_t u64() { uint64_t v = 0; pod(v); return v; }
    uint64_t count() {
        const uint64_t n = u64();
        if (n > kMaxItems) ok = false;
        return ok ? n : 0;
    }
    void str(std::string& s) { s.resize(count()); raw(s.data(), s.size()); }
    template<class T> void vec(std::vector<T>& v) { v.resize(count()); raw(v.data(), v.size() * sizeof(T)); }
    void buf(ConversationBuffer& b) {
        b = {};
        b.resize(count());
        if (!b.visit(0, b.size(), [&](uint8_t* p, size_t n, size_t) { raw(p, n); return ok; })) ok = false;
    }
};

void put_checkpoint(Writer& w, const ConversationCheckpoint& c) {
    w.vec(c.ids);
    w.vec(c.imgs);
    w.vec(c.gdn); w.vec(c.ple); w.vec(c.tails); w.vec(c.dead); w.vec(c.block_pos);
    w.u64(c.used);
    w.u64(c.stage_parts.size());
    for (const auto& p : c.stage_parts) put_checkpoint(w, p);
}

void get_checkpoint(Reader& r, ConversationCheckpoint& c) {
    r.vec(c.ids);
    r.vec(c.imgs);
    r.vec(c.gdn); r.vec(c.ple); r.vec(c.tails); r.vec(c.dead); r.vec(c.block_pos);
    c.used = r.u64();
    c.stage_parts.resize(r.count());
    for (auto& p : c.stage_parts) { if (!r.ok) return; get_checkpoint(r, p); }
}

void put_kv(Writer& w, const ConversationKv& k) {
    w.pod(k.format);
    w.pod(k.cells); w.pod(k.heads); w.pod(k.head_dim); w.pod(k.page_size); w.pod(k.pooled_rows); w.pod(k.idx_dim);
    w.buf(k.k); w.buf(k.v); w.buf(k.k_scale); w.buf(k.v_scale); w.buf(k.pooled);
}

void get_kv(Reader& r, ConversationKv& k) {
    r.pod(k.format);
    r.pod(k.cells); r.pod(k.heads); r.pod(k.head_dim); r.pod(k.page_size); r.pod(k.pooled_rows); r.pod(k.idx_dim);
    r.buf(k.k); r.buf(k.v); r.buf(k.k_scale); r.buf(k.v_scale); r.buf(k.pooled);
}

void put_image(Writer& w, const SavedConversation& s) {
    w.pod(s.geometry);
    w.pod(s.layer_lo); w.pod(s.layer_hi);
    put_checkpoint(w, s.live);
    w.u64(s.checkpoints.size());
    for (const auto& c : s.checkpoints) put_checkpoint(w, c);
    w.u64(s.kv.size());
    for (const auto& k : s.kv) put_kv(w, k);
    w.pod(s.cvec);
    w.u64(s.stage_images.size());
    for (const auto& st : s.stage_images) put_image(w, st);
}

void get_image(Reader& r, SavedConversation& s) {
    r.pod(s.geometry);
    r.pod(s.layer_lo); r.pod(s.layer_hi);
    get_checkpoint(r, s.live);
    s.checkpoints.resize(r.count());
    for (auto& c : s.checkpoints) { if (!r.ok) return; get_checkpoint(r, c); }
    s.kv.resize(r.count());
    for (auto& k : s.kv) { if (!r.ok) return; get_kv(r, k); }
    r.pod(s.cvec);
    s.stage_images.resize(r.count());
    for (auto& st : s.stage_images) { if (!r.ok) return; get_image(r, st); }
}

void put_identity(Writer& w, const ConversationFileIdentity& id) {
    w.str(id.model_path);
    w.u64(id.model_bytes);
    w.pod(id.model_mtime_ns);
    w.str(id.engine);
}

}  // namespace

bool conversation_file_identity(const std::string& model_path, const std::string& engine,
                                ConversationFileIdentity& out, std::string& err) {
    struct stat st {};
    if (::stat(model_path.c_str(), &st) != 0) {
        err = model_path + ": " + std::strerror(errno);
        return false;
    }
    out.model_path = model_path;
    out.model_bytes = (uint64_t) st.st_size;
    out.model_mtime_ns = (int64_t) st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
    out.engine = engine;
    return true;
}

bool conversation_file_write(const std::string& path, const ConversationFileIdentity& id,
                             const std::vector<const SavedConversation*>& images, size_t& bytes, std::string& err) {
    bytes = 0;
    const std::string tmp = path + ".tmp";
    {
        File f(std::fopen(tmp.c_str(), "wb"));
        if (!f) { err = tmp + ": " + std::strerror(errno); return false; }
        std::setvbuf(f.get(), nullptr, _IOFBF, 8 << 20);
        Writer w{f.get()};
        w.raw(kMagic, sizeof kMagic);
        w.pod(kVersion);
        put_identity(w, id);
        w.u64(images.size());
        for (const auto* s : images) put_image(w, *s);
        w.raw(kTail, sizeof kTail);
        if (std::fflush(f.get()) != 0) w.ok = false;
        bytes = w.bytes;
        if (!w.ok) {
            err = tmp + ": write failed (" + std::strerror(errno) + ")";
            f.reset();
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = path + ": " + std::strerror(errno);
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool conversation_file_read(const std::string& path, const ConversationFileIdentity& id,
                            std::vector<SavedConversation>& out, size_t& bytes, std::string& err) {
    out.clear();
    bytes = 0;
    File f(std::fopen(path.c_str(), "rb"));
    if (!f) { err = path + ": " + std::strerror(errno); return false; }
    std::setvbuf(f.get(), nullptr, _IOFBF, 8 << 20);
    Reader r{f.get()};
    char magic[sizeof kMagic] = {};
    r.raw(magic, sizeof magic);
    uint32_t version = 0;
    r.pod(version);
    if (!r.ok || std::memcmp(magic, kMagic, sizeof kMagic) != 0 || version != kVersion) {
        err = path + ": not a conversation file of this version";
        return false;
    }
    ConversationFileIdentity got;
    r.str(got.model_path);
    got.model_bytes = r.u64();
    r.pod(got.model_mtime_ns);
    r.str(got.engine);
    if (!r.ok) { err = path + ": damaged header"; return false; }
    if (got.model_path != id.model_path || got.model_bytes != id.model_bytes || got.model_mtime_ns != id.model_mtime_ns) {
        err = path + ": made with another model file (" + got.model_path + ")";
        return false;
    }
    if (got.engine != id.engine) {
        err = path + ": made by another engine build (" + got.engine + ", this is " + id.engine + ")";
        return false;
    }
    std::vector<SavedConversation> images(r.count());
    for (auto& s : images) { if (!r.ok) break; get_image(r, s); }
    char tail[sizeof kTail] = {};
    r.raw(tail, sizeof tail);
    if (!r.ok || std::memcmp(tail, kTail, sizeof kTail) != 0) {
        err = path + ": damaged or truncated";
        return false;
    }
    bytes = r.bytes;
    out = std::move(images);
    return true;
}

}  // namespace basalt::core
