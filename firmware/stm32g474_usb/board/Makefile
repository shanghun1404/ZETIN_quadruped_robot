# arm-none-eabi build of the real AK firmware. VENDOR holds the ST HAL,
# CMSIS device and core headers (same layout as the 2026-10-20 fixture).
CC = arm-none-eabi-gcc
VENDOR ?= vendor
CORE ?= ..
BUILD ?= build
CPU = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CPPFLAGS = -DSTM32G474xx -DUSE_HAL_DRIVER -I. -I$(CORE)/include -I$(CORE)/examples -I$(VENDOR)/hal/Inc -I$(VENDOR)/device/Include -I$(VENDOR)/core
CFLAGS = $(CPU) -std=c11 -O2 -g3 -Wall -Wextra -Werror -ffunction-sections -fdata-sections
HAL_NAMES = hal hal_cortex hal_rcc hal_rcc_ex hal_flash hal_flash_ex hal_pwr hal_pwr_ex hal_gpio hal_uart hal_uart_ex hal_dma hal_dma_ex hal_fdcan
SOURCES = g474_ak_firmware.c g474_ak_config.c $(CORE)/src/qd_protocol.c $(CORE)/src/qd_controller.c $(CORE)/examples/qd_ak_motor.c $(VENDOR)/device/Source/Templates/system_stm32g4xx.c $(addprefix $(VENDOR)/hal/Src/stm32g4xx_,$(addsuffix .c,$(HAL_NAMES)))
OBJECTS = $(addprefix $(BUILD)/,$(notdir $(SOURCES:.c=.o))) $(BUILD)/startup.o
vpath %.c $(sort $(dir $(SOURCES)))

.PHONY: all
all: $(BUILD)/g474_ak_firmware.elf $(BUILD)/g474_ak_firmware.bin
$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: %.c g474_ak_diag.h $(CORE)/examples/qd_ak_motor.h $(CORE)/include/qd_protocol.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
$(BUILD)/startup.o: $(VENDOR)/device/Source/Templates/gcc/startup_stm32g474xx.s | $(BUILD)
	$(CC) $(CPU) -g3 -c $< -o $@
$(BUILD)/g474_ak_firmware.elf: $(OBJECTS) g474.ld
	$(CC) $(CPU) $(OBJECTS) -Tg474.ld --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections,-Map=$(BUILD)/g474_ak_firmware.map -Wl,--print-memory-usage -lm -o $@
	arm-none-eabi-size $@
$(BUILD)/g474_ak_firmware.bin: $(BUILD)/g474_ak_firmware.elf
	arm-none-eabi-objcopy -O binary $< $@
