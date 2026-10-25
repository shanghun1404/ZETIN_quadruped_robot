# Bench-only CAN timing fixture. BOARD supplies stm32g4xx_hal_conf.h and g474.ld
# of the AK firmware; VENDOR the ST HAL/CMSIS (same layout as board/Makefile).
CC = arm-none-eabi-gcc
VENDOR ?= vendor
BOARD ?= ../../../../firmware/stm32g474_usb/board
CAN_KBPS ?= 250
BUILD ?= build$(CAN_KBPS)
CPU = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CPPFLAGS = -DSTM32G474xx -DUSE_HAL_DRIVER -DG474_CAN_KBPS=$(CAN_KBPS)u -I$(BOARD) -I$(VENDOR)/hal/Inc -I$(VENDOR)/device/Include -I$(VENDOR)/core
CFLAGS = $(CPU) -std=c11 -O2 -g3 -Wall -Wextra -Werror -ffunction-sections -fdata-sections
HAL_NAMES = hal hal_cortex hal_rcc hal_rcc_ex hal_flash hal_flash_ex hal_pwr hal_pwr_ex hal_gpio hal_fdcan
SOURCES = g474_can_bench.c $(VENDOR)/device/Source/Templates/system_stm32g4xx.c $(addprefix $(VENDOR)/hal/Src/stm32g4xx_,$(addsuffix .c,$(HAL_NAMES)))
OBJECTS = $(addprefix $(BUILD)/,$(notdir $(SOURCES:.c=.o))) $(BUILD)/startup.o
vpath %.c $(sort $(dir $(SOURCES)))

.PHONY: all
all: $(BUILD)/g474_can_bench.elf
$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: %.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
$(BUILD)/startup.o: $(VENDOR)/device/Source/Templates/gcc/startup_stm32g474xx.s | $(BUILD)
	$(CC) $(CPU) -g3 -c $< -o $@
$(BUILD)/g474_can_bench.elf: $(OBJECTS) $(BOARD)/g474.ld
	$(CC) $(CPU) $(OBJECTS) -T$(BOARD)/g474.ld --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections -Wl,--print-memory-usage -o $@
	arm-none-eabi-size $@
