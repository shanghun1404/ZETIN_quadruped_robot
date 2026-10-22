"""Optional scratch boundary timing; delegates actual operations unchanged."""
import time
from v14_ros2_control.contract import stamp_ns
from logging_buffers import NumericBuffer


def instrument(node, output, seconds=3600):
    phases = ('states', 'odoms', 'images', 'contacts', 'publish')
    trace = NumericBuffer(round((seconds + 30) * 400) + 1000,
        [('phase', 'i1'), ('stamp_ns', 'i8'), ('start_ns', 'i8'),
         ('end_ns', 'i8'), ('thread_cpu_ms', 'f8')])
    original_add = node.add
    def add(buffer, message):
        start = time.perf_counter_ns();cpu = time.thread_time_ns()
        phase = next(i for i, name in enumerate(phases[:4]) if buffer is getattr(node, name))
        try:
            return original_add(buffer, message)
        finally:
            trace.append((phase, stamp_ns(message.header.stamp), start,
                time.perf_counter_ns(), (time.thread_time_ns() - cpu) / 1e6))
    node.add = add
    original_pub = node.plan_pub
    class Publisher:
        def publish(self, message):
            start = time.perf_counter_ns();cpu = time.thread_time_ns()
            try:
                return original_pub.publish(message)
            finally:
                trace.append((4, node.last, start, time.perf_counter_ns(),
                    (time.thread_time_ns() - cpu) / 1e6))
    node.plan_pub = Publisher()
    original_destroy = node.destroy_node
    def destroy():
        try:
            return original_destroy()
        finally:
            trace.write_csv(output / 'planner_input_timing.csv',
                convert=lambda row: [phases[row[0]], *row[1:]])
    node.destroy_node = destroy
