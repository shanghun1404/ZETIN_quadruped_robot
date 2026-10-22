CC = arm-none-eabi-gcc
CPU = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CPPFLAGS = -DSTM32G474xx -DUSE_HAL_DRIVER -Ifirmware -Icore/include -Icore/examples -Ivendor/hal/Inc -Ivendor/device/Include -Ivendor/core
CFLAGS = $(CPU) -std=c11 -O2 -g3 -Wall -Wextra -Werror -ffunction-sections -fdata-sections
HAL_NAMES = hal hal_cortex hal_rcc hal_rcc_ex hal_flash hal_flash_ex hal_pwr hal_pwr_ex hal_gpio hal_uart hal_uart_ex hal_dma hal_dma_ex hal_fdcan
SOURCES = firmware/g474_can_fixture.c core/src/qd_protocol.c core/src/qd_controller.c core/examples/qd_ak_virtual.c vendor/device/Source/Templates/system_stm32g4xx.c $(addprefix vendor/hal/Src/stm32g4xx_,$(addsuffix .c,$(HAL_NAMES)))
OBJECTS = $(addprefix build/,$(SOURCES:.c=.o)) build/startup.o
.PHONY: all
all: build/g474_can_fixture.elf build/g474_can_fixture.bin
build/%.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
build/startup.o: vendor/device/Source/Templates/gcc/startup_stm32g474xx.s
	mkdir -p $(dir $@)
	$(CC) $(CPU) -g3 -c $< -o $@
build/g474_can_fixture.elf: $(OBJECTS) firmware/g474.ld
	$(CC) $(CPU) $(OBJECTS) -Tfirmware/g474.ld --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections,-Map=build/g474_can_fixture.map -Wl,--print-memory-usage -lm -o $@
	arm-none-eabi-size $@
build/g474_can_fixture.bin: build/g474_can_fixture.elf
	arm-none-eabi-objcopy -O binary $< $@
