CC ?= cc
CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Werror -pedantic
CPPFLAGS += -Iinclude
LDLIBS += -lm
CORE = src/qd_protocol.c src/qd_controller.c
BUILD ?= build

.PHONY: all test clean
all: $(BUILD)/qd_emulator $(BUILD)/libqd_protocol.so
$(BUILD):
	mkdir -p $@
$(BUILD)/test_protocol: tests/test_protocol.c $(CORE) include/qd_protocol.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_protocol.c $(CORE) $(LDLIBS) -o $@
$(BUILD)/test_g474_port: tests/test_g474_port.c tests/hal_stub/main.h examples/g474_hal_port.c examples/qd_g474_port.h $(CORE) include/qd_protocol.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Iexamples -Itests/hal_stub $(CFLAGS) tests/test_g474_port.c $(CORE) $(LDLIBS) -o $@
$(BUILD)/test_can_diagnostic: tests/test_can_diagnostic.c examples/qd_can_diagnostic.c examples/qd_can_diagnostic.h include/qd_protocol.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Iexamples $(CFLAGS) tests/test_can_diagnostic.c examples/qd_can_diagnostic.c $(LDLIBS) -o $@
$(BUILD)/test_ak_virtual: tests/test_ak_virtual.c examples/qd_ak_virtual.c examples/qd_ak_virtual.h include/qd_protocol.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Iexamples $(CFLAGS) tests/test_ak_virtual.c examples/qd_ak_virtual.c $(LDLIBS) -o $@
$(BUILD)/qd_emulator: tools/qd_emulator.c $(CORE) include/qd_protocol.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) tools/qd_emulator.c $(CORE) $(LDLIBS) -o $@
$(BUILD)/libqd_protocol.so: $(CORE) include/qd_protocol.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared $(CORE) $(LDLIBS) -o $@
test: $(BUILD)/test_protocol $(BUILD)/test_g474_port $(BUILD)/test_can_diagnostic $(BUILD)/test_ak_virtual
	$(abspath $(BUILD))/test_protocol
	$(abspath $(BUILD))/test_g474_port
	$(abspath $(BUILD))/test_can_diagnostic
	$(abspath $(BUILD))/test_ak_virtual
clean:
	rm -rf $(BUILD)
