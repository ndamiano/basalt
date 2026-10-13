// Parked conversations on disk: the --serve conversation cache written to one file and read back by a later engine
// process, so a restart (an unload around an image generation, say) resumes a long conversation from its K/V and
// running state instead of reading the whole prompt again.
//
// The file is bound to the model file it was made with (path, size, modification time) and to the engine build: a
// file from anything else is refused as a whole.  The images themselves are then checked as they are restored
// (conversation_snapshot_validate), exactly as an image parked in RAM is, so a geometry or K/V-format mismatch
// discards that image and the request reads its prompt normally.
#pragma once

#include "basalt/core/conversation_cache.hpp"

#include <string>
#include <vector>

namespace basalt::core {

struct ConversationFileIdentity {
    std::string model_path;
    uint64_t model_bytes = 0;
    int64_t model_mtime_ns = 0;
    std::string engine;            ///< the engine build (BASALT_VERSION)
};

/// The identity of `model_path` as it is on disk now; false (with `err`) when it cannot be read.
bool conversation_file_identity(const std::string& model_path, const std::string& engine,
                                ConversationFileIdentity& out, std::string& err);

/// Write `images` to `path` (through a temporary file renamed over it, so a reader never sees half a file).
bool conversation_file_write(const std::string& path, const ConversationFileIdentity& id,
                             const std::vector<const SavedConversation*>& images, size_t& bytes, std::string& err);

/// Read `path` into `out`; false (with `err`, `out` empty) when it is missing, damaged or made with another model or
/// engine build.
bool conversation_file_read(const std::string& path, const ConversationFileIdentity& id,
                            std::vector<SavedConversation>& out, size_t& bytes, std::string& err);

}  // namespace basalt::core
