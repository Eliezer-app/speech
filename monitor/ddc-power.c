// ddc-power — Turn external monitor on/off via DDC/CI on Apple Silicon
// Build: clang -o ddc-power ddc-power.c -framework IOKit -framework CoreFoundation
// Usage: ddc-power on        (0x01 — wake)
//        ddc-power standby   (0x02 — low power, quick wake)
//        ddc-power suspend   (0x03 — lower power, slower wake)
//        ddc-power off       (0x04 — soft off, DDC responsive)
//        ddc-power hardoff   (0x05 — hard off, may need replug)

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <stdio.h>
#include <string.h>

// Private IOAVService API
typedef CFTypeRef IOAVServiceRef;
extern IOAVServiceRef IOAVServiceCreate(CFAllocatorRef allocator);
extern IOReturn IOAVServiceWriteI2C(IOAVServiceRef service, uint32_t chipAddress, uint32_t dataAddress, void *inputBuffer, uint32_t inputBufferSize);

// DDC/CI VCP code for power mode
#define VCP_POWER 0xD6
#define POWER_ON       0x01
#define POWER_STANDBY  0x02
#define POWER_SUSPEND  0x03
#define POWER_OFF      0x04
#define POWER_HARD_OFF 0x05

static int ddc_set_vcp(IOAVServiceRef service, uint8_t code, uint16_t value) {
    uint8_t data[6];
    data[0] = 0x84;              // length: 4 bytes follow + 0x80 flag
    data[1] = 0x03;              // set VCP command
    data[2] = code;              // VCP code
    data[3] = (value >> 8) & 0xFF; // value high byte
    data[4] = value & 0xFF;       // value low byte
    // checksum: XOR of destination (0x6E), source flag (0x51), and all data bytes
    data[5] = 0x6E ^ 0x51;
    for (int i = 0; i < 5; i++) data[5] ^= data[i];

    IOReturn ret = IOAVServiceWriteI2C(service, 0x37, 0x51, data, 6);
    return ret == kIOReturnSuccess ? 0 : -1;
}

int main(int argc, char *argv[]) {
    if (argc != 2) goto usage;

    uint16_t value;
    const char *label;
    if      (strcmp(argv[1], "on")      == 0) { value = POWER_ON;       label = "on"; }
    else if (strcmp(argv[1], "standby") == 0) { value = POWER_STANDBY;  label = "standby"; }
    else if (strcmp(argv[1], "suspend") == 0) { value = POWER_SUSPEND;  label = "suspend"; }
    else if (strcmp(argv[1], "off")     == 0) { value = POWER_OFF;      label = "off"; }
    else if (strcmp(argv[1], "hardoff") == 0) { value = POWER_HARD_OFF; label = "hard off"; }
    else goto usage;

    IOAVServiceRef service = IOAVServiceCreate(kCFAllocatorDefault);
    if (!service) {
        fprintf(stderr, "No display found (IOAVServiceCreate failed)\n");
        return 1;
    }

    if (ddc_set_vcp(service, VCP_POWER, value) != 0) {
        fprintf(stderr, "DDC write failed\n");
        CFRelease(service);
        return 1;
    }

    printf("Display %s\n", label);
    CFRelease(service);
    return 0;

usage:
    fprintf(stderr,
        "Usage: ddc-power <command>\n"
        "\n"
        "Commands:\n"
        "  on       0x01 — Wake display\n"
        "  standby  0x02 — Low power, quick wake\n"
        "  suspend  0x03 — Lower power, slower wake\n"
        "  off      0x04 — Soft off, DDC still responsive\n"
        "  hardoff  0x05 — Hard off, may need cable replug\n"
    );
    return 1;
}
