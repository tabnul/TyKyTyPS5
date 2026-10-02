#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUCHECKPOINTS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUCHECKPOINTS_H_

// KYTY_GPU_CHECKPOINTS=1 (diagnostic, VK_NV_device_diagnostic_checkpoints): a checkpoint at each
// GpuZones mark (every guest draw and dispatch, keyed by shader hash, and the emulator's own
// copies, blits and tiling), read back after a lost device to name the work the GPU hung on.
// It takes the GpuZones marker, so KYTY_GPU_ZONES has no effect while it is on.
namespace Libs::Graphics {
struct GraphicContext;
} // namespace Libs::Graphics

namespace Libs::Graphics::GpuCheckpoints {

// Installs the marker when the device was created with the extension.
void Install(GraphicContext& graphics);
// Logs the last checkpoints the queue reached; call only after VK_ERROR_DEVICE_LOST.
void Report(GraphicContext& graphics);

} // namespace Libs::Graphics::GpuCheckpoints

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUCHECKPOINTS_H_
