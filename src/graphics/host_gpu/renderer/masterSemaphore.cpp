#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuCheckpoints.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::RecordSubmit(const SubmitRecord& record) {
	std::lock_guard lock(m_record_mutex);
	m_records[record.tick % RecordSlots] = record;
}

void MasterSemaphore::Fail(const char* what, vk::Result result, uint64_t tick) {
	// Every thread waiting on the GPU sees a lost device; the first to get here reports it and
	// exits, and the others wait for that.
	static std::mutex fail_mutex;
	fail_mutex.lock();
	const auto gpu_tick = KnownGpuTick();
	const auto current  = CurrentTick();
	LOGF("%s failed: %s (%d), waited tick=%" PRIu64 " gpu tick=%" PRIu64 " next tick=%" PRIu64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, gpu_tick, current);
	std::printf("%s failed: %s (%d), waited tick=%" PRIu64 " gpu tick=%" PRIu64
	            " next tick=%" PRIu64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, gpu_tick,
	            current);
	{
		// The first unfinished submission is the likeliest one to have hung or faulted the GPU.
		std::lock_guard lock(m_record_mutex);
		const uint64_t  oldest = current > RecordSlots ? current - RecordSlots : 1;
		// A lost device can report any counter value (NVIDIA: UINT64_MAX); list every record then.
		const auto      first  = gpu_tick < current ? std::max<uint64_t>(gpu_tick + 1, oldest) : oldest;
		for (auto t = first; t < current; ++t) {
			const auto& record = m_records[t % RecordSlots];
			if (record.tick != t) {
				continue;
			}
			LOGF("  in flight: tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64 " pm4_op=0x%x"
			     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
			     t, record.debug_op, record.debug_submit, record.pm4_op, record.debug_arg0,
			     record.debug_arg1, record.debug_arg2, record.debug_arg3, record.debug_arg4);
			std::printf("  in flight: tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
			            " pm4_op=0x%x args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
			            t, record.debug_op, record.debug_submit, record.pm4_op, record.debug_arg0,
			            record.debug_arg1, record.debug_arg2, record.debug_arg3, record.debug_arg4);
		}
	}
	if (result == vk::Result::eErrorDeviceLost) {
		GpuCheckpoints::Report(m_graphics);
	}
	std::fflush(stdout);
	EXIT("%s failed: %s\n", what, vk::to_string(result).c_str());
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess) {
		Fail("vkGetSemaphoreCounterValue", result, KnownGpuTick() + 1);
	}

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	if (result != vk::Result::eSuccess) {
		Fail("vkWaitSemaphores", result, tick);
	}
	Refresh();
}

} // namespace Libs::Graphics
