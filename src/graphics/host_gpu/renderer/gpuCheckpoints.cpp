#include "graphics/host_gpu/renderer/gpuCheckpoints.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuZones.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <vector>

namespace Libs::Graphics::GpuCheckpoints {

namespace {

struct Entry {
	uint64_t         id     = 0;
	DrainStats::Zone zone   = DrainStats::Zone::Unmarked;
	uint64_t         key    = 0;
	uint64_t         pixels = 0;
};

constexpr uint64_t Slots = uint64_t {1} << 16u;
// Context lines printed around the reported checkpoints.
constexpr uint64_t Context = 4;
constexpr uint64_t MaxLines = 64;

std::mutex         g_mutex;
std::vector<Entry> g_entries;
uint64_t           g_next = 1;

// Marks come from every scheduler's recording thread; the queue orders the checkpoints.
void Mark(void* /*context*/, vk::CommandBuffer buffer, DrainStats::Zone zone, uint64_t key,
          uint64_t pixels) {
	uint64_t id = 0;
	{
		std::lock_guard lock(g_mutex);
		id                     = g_next++;
		g_entries[id % Slots] = {id, zone, key, pixels};
	}
	buffer.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(id)));
}

template <typename... Args>
void Print(const char* format, Args... args) {
	LOGF(format, args...);
	std::printf(format, args...);
}

void PrintEntry(const char* prefix, uint64_t id) {
	const auto& entry = g_entries[id % Slots];
	if (entry.id != id) {
		Print("%s#%" PRIu64 " (overwritten)\n", prefix, id);
		return;
	}
	Print("%s#%" PRIu64 " %s key=0x%016" PRIx64 " pixels=%" PRIu64 "\n", prefix, id,
	      DrainStats::ZoneLabel(entry.zone), entry.key, entry.pixels);
}

} // namespace

void Install(GraphicContext& graphics) {
	if (!graphics.checkpoints_enabled || GpuZones::g_marker != nullptr) {
		return;
	}
	g_entries.resize(Slots);
	GpuZones::g_context = nullptr;
	GpuZones::g_marker  = &Mark;
	Print("GPU checkpoints: on\n");
}

void ReportFault(GraphicContext& graphics) {
	if (!graphics.device_fault_enabled ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT == nullptr) {
		return;
	}
	vk::DeviceFaultCountsEXT counts {};
	if (graphics.device.getFaultInfoEXT(&counts, nullptr) != vk::Result::eSuccess) {
		Print("GPU fault: no information\n");
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendor(counts.vendorInfoCount);
	counts.vendorBinarySize = 0;
	vk::DeviceFaultInfoEXT info {};
	info.pAddressInfos = addresses.data();
	info.pVendorInfos  = vendor.data();
	const auto result  = graphics.device.getFaultInfoEXT(&counts, &info);
	Print("GPU fault (%s): %s\n", vk::to_string(result).c_str(), info.description.data());
	for (uint32_t i = 0; i < counts.addressInfoCount; ++i) {
		const auto& address = addresses[i];
		Print("  %s address=0x%016" PRIx64 " precision=0x%" PRIx64 "\n",
		      vk::to_string(address.addressType).c_str(), address.reportedAddress,
		      address.addressPrecision);
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; ++i) {
		Print("  vendor: %s code=0x%" PRIx64 " data=0x%" PRIx64 "\n", vendor[i].description.data(),
		      vendor[i].vendorFaultCode, vendor[i].vendorFaultData);
	}
}

void Report(GraphicContext& graphics) {
	ReportFault(graphics);
	if (!graphics.checkpoints_enabled || GpuZones::g_marker != &Mark ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV == nullptr) {
		return;
	}
	uint32_t count = 0;
	graphics.queue.getCheckpointDataNV(&count, nullptr);
	std::vector<vk::CheckpointDataNV> data(count);
	graphics.queue.getCheckpointDataNV(&count, data.data());
	data.resize(count);

	std::lock_guard lock(g_mutex);
	Print("GPU checkpoints: %" PRIu64 " recorded, %u reported by the queue\n", g_next - 1, count);
	if (count == 0) {
		return;
	}
	uint64_t low  = UINT64_MAX;
	uint64_t high = 0;
	for (const auto& checkpoint : data) {
		const auto id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(checkpoint.pCheckpointMarker));
		Print("  %s:\n", vk::to_string(checkpoint.stage).c_str());
		PrintEntry("    last reached ", id);
		low  = std::min(low, id);
		high = std::max(high, id);
	}
	// Work recorded between the checkpoints the pipeline's ends reached, and just after: the GPU
	// was executing (or stuck in) these.
	const auto first = low > Context ? low - Context : 1;
	const auto last  = std::min({high + Context, g_next - 1, first + MaxLines - 1});
	Print("  checkpoints %" PRIu64 "..%" PRIu64 ":\n", first, last);
	for (auto id = first; id <= last; ++id) {
		PrintEntry(id == low || id == high ? "  > " : "    ", id);
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::GpuCheckpoints
