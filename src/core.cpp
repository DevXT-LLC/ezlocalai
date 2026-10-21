#define _POSIX_C_SOURCE 200809L

#include "ezlocalai/ezlocalai_c.h"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/tts_text.hpp"
#include <algorithm>
#include <new>
#include <stdexcept>

#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static std::once_flag dotenv_once;

static const char *default_value_for(const char *name) {
    if (!name) return "";
    for (const auto &entry : ezlocalai::environment_defaults) {
        if (entry.first == name) return entry.second.data();
    }
    return "";
}

static void load_dotenv_once(void) {
    (void)ezl_load_dotenv();
}

static void ensure_dotenv_loaded(void) {
    std::call_once(dotenv_once, load_dotenv_once);
}

static void copy_env(char *out, size_t out_size, const char *name) {
    ensure_dotenv_loaded();
    const char *value = getenv(name);
    if (!value) {
        value = default_value_for(name);
    }
    snprintf(out, out_size, "%s", value);
}

static size_t parse_size_value(const char *value, size_t fallback) {
    if (!value || value[0] == '\0') {
        return fallback;
    }
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (end == value || parsed == 0) {
        return fallback;
    }
    return (size_t)parsed;
}

static size_t parse_size_env(const char *name, size_t fallback) {
    ensure_dotenv_loaded();
    const char *value = getenv(name);
    if (!value || value[0] == '\0') {
        value = default_value_for(name);
    }
    return parse_size_value(value, fallback);
}

static long parse_seconds_env_ms(const char *name, long fallback_ms) {
    ensure_dotenv_loaded();
    const char *value = getenv(name);
    if (!value || value[0] == '\0') {
        value = default_value_for(name);
    }
    char *end = NULL;
    double parsed = strtod(value, &end);
    if (end == value || parsed <= 0.0) {
        return fallback_ms;
    }
    return (long)(parsed * 1000.0);
}

const char *ezl_getenv_default(const char *name) {
    return ezl_getenv_fallback(name, nullptr);
}

const char *ezl_getenv_fallback(const char *name, const char *fallback) {
    ensure_dotenv_loaded();
    if (!name) return fallback && *fallback ? fallback : "";
    const char *value = getenv(name);
    if (value) {
        return value;
    }
    return fallback && *fallback ? fallback : default_value_for(name);
}

void ezl_load_config(ezl_env_config *out) {
    if (!out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    copy_env(out->url, sizeof(out->url), "EZLOCALAI_URL");
    copy_env(out->api_key, sizeof(out->api_key), "EZLOCALAI_API_KEY");
    copy_env(out->default_model, sizeof(out->default_model), "DEFAULT_MODEL");
    copy_env(out->log_level, sizeof(out->log_level), "LOG_LEVEL");
    out->max_queue_size = parse_size_env("MAX_QUEUE_SIZE", 100);
    const char *n_parallel = getenv("N_PARALLEL");
    if (n_parallel && n_parallel[0] != '\0') {
        out->max_concurrent_requests = parse_size_value(n_parallel, 1);
    } else {
        out->max_concurrent_requests = parse_size_env("MAX_CONCURRENT_REQUESTS", 1);
    }
    out->request_timeout_ms = parse_seconds_env_ms("REQUEST_TIMEOUT", 120000);
}

static int format_timestamp(double seconds, char separator, char *out, size_t out_size) {
    if (!out || out_size == 0 || seconds < 0.0) {
        return EINVAL;
    }
    int hours = (int)(seconds / 3600.0);
    int minutes = (int)((seconds - (hours * 3600)) / 60.0);
    int secs = (int)seconds % 60;
    int whole = (int)seconds;
    int millis = (int)((seconds - whole) * 1000.0);
    if (millis < 0) {
        millis = 0;
    }
    int written = snprintf(out, out_size, "%02d:%02d:%02d%c%03d", hours, minutes, secs, separator, millis);
    if (written < 0 || (size_t)written >= out_size) {
        return ENOSPC;
    }
    return 0;
}

int ezl_format_timestamp_srt(double seconds, char *out, size_t out_size) {
    return format_timestamp(seconds, ',', out, out_size);
}

int ezl_format_timestamp_vtt(double seconds, char *out, size_t out_size) {
    return format_timestamp(seconds, '.', out, out_size);
}

static char *ezl_strdup_range(const char *begin, size_t len) {
    char *out = (char *)malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, begin, len);
    out[len] = '\0';
    return out;
}

static char *ezl_strdup_c(const char *value) {
    return ezl_strdup_range(value ? value : "", strlen(value ? value : ""));
}

static int list_append(ezl_string_list *list, const char *value) {
    char **new_items = (char **)realloc(list->items, sizeof(char *) * (list->count + 1));
    if (!new_items) {
        return ENOMEM;
    }
    list->items = new_items;
    list->items[list->count] = ezl_strdup_c(value);
    if (!list->items[list->count]) {
        return ENOMEM;
    }
    list->count++;
    return 0;
}

void ezl_string_list_free(ezl_string_list *list) {
    if (!list) {
        return;
    }
    for (size_t i = 0; i < list->count; ++i) {
        free(list->items[i]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0;
}

static int copy_tts_result(const std::string &value,char *out,size_t size) {
    const auto bytes=std::min(size-1,value.size()); memcpy(out,value.data(),bytes); out[bytes]='\0'; return bytes==value.size()?0:ENOSPC;
}
int ezl_number_to_words(long n,char *out,size_t size) {
    if(!out || !size) return EINVAL;
    try { return copy_tts_result(ezlocalai::tts_number_to_words(std::to_string(n)),out,size); } catch(const std::bad_alloc &) { return ENOMEM; } catch(const std::exception &) { return EINVAL; }
}
int ezl_normalize_text_for_tts(const char *text,char *out,size_t size) {
    if(!text || !out || !size) return EINVAL;
    try { return copy_tts_result(ezlocalai::tts_normalize_english(text),out,size); } catch(const std::bad_alloc &) { return ENOMEM; } catch(const std::exception &) { return EINVAL; }
}
int ezl_split_text_into_chunks(const char *text,size_t maximum,ezl_string_list *out) {
    if(!text || !out) return EINVAL; memset(out,0,sizeof(*out));
    try { for(const auto &chunk:ezlocalai::tts_split_text(text,static_cast<double>(maximum))) { const auto result=list_append(out,chunk.c_str()); if(result) { ezl_string_list_free(out); return result; } } return 0; }
    catch(const std::bad_alloc &) { ezl_string_list_free(out); return ENOMEM; } catch(const std::exception &) { ezl_string_list_free(out); return EINVAL; }
}

int ezl_gpu_tier_for_name(const char *name) {
    static const struct {
        const char *needle;
        int tier;
    } tiers[] = {
        {"h200", 100},
        {"h100", 95},
        {"b200", 110},
        {"gb10", 50},
        {"a100", 80},
        {"a6000", 70},
        {"a40", 65},
        {"l40", 70},
        {"5090", 90},
        {"5080", 75},
        {"5070 ti", 60},
        {"5070", 55},
        {"5060", 40},
        {"4090", 80},
        {"4080 super", 65},
        {"4080", 60},
        {"4070 ti super", 55},
        {"4070 ti", 50},
        {"4070", 45},
        {"4060 ti", 35},
        {"4060", 30},
        {"3090 ti", 55},
        {"3090", 50},
        {"3080 ti", 45},
        {"3080", 40},
        {"3070 ti", 32},
        {"3070", 30},
        {"3060 ti", 25},
        {"3060", 22},
        {"2080 ti", 28},
        {"2080", 22},
        {"2070", 18},
        {"2060", 15},
        {"1080 ti", 14},
        {"1080", 12},
        {"1070", 10},
        {"agx orin", 35},
        {"orin nx 16g", 25},
        {"orin nx", 22},
        {"orin nano", 12},
        {"orin", 20},
        {"agx xavier", 16},
        {"xavier nx 16g", 14},
        {"xavier nx", 12},
        {"xavier", 10},
        {"tx2", 6},
        {"nano", 4},
        {"mi300x", 90},
        {"mi300", 85},
        {"mi250x", 70},
        {"mi250", 65},
        {"mi210", 55},
        {"mi100", 45},
        {"rx 7900 xtx", 55},
        {"rx 7900 xt", 50},
        {"rx 7900", 48},
        {"rx 7800", 38},
        {"rx 7700", 30},
        {"rx 7600", 22},
        {"rx 6950", 42},
        {"rx 6900", 38},
        {"rx 6800", 32},
        {"rx 6700", 25},
        {"rx 6600", 18},
        {"radeon 780m", 8},
        {"radeon 760m", 6},
        {"radeon 680m", 5},
        {"radeon", 4},
        {"m4 max", 50},
        {"m4 ultra", 65},
        {"m4 pro", 35},
        {"m4", 25},
        {"m3 ultra", 45},
        {"m3 max", 38},
        {"m3 pro", 28},
        {"m3", 20},
        {"m2 ultra", 40},
        {"m2 max", 32},
        {"m2 pro", 22},
        {"m2", 16},
        {"m1 ultra", 30},
        {"m1 max", 25},
        {"m1 pro", 18},
        {"m1", 12},
        {"hailo-8l", 6},
        {"hailo-8", 8},
        {"hailo", 5},
        {"arc a770", 18},
        {"arc a750", 14},
        {"arc a380", 8},
        {"arc", 6},
        {"xe", 4},
    };
    if (!name || name[0] == '\0') {
        return 2;
    }
    size_t len = strlen(name);
    char *lowered = (char *)malloc(len + 1);
    if (!lowered) {
        return 20;
    }
    for (size_t i = 0; i < len; ++i) {
        lowered[i] = (char)tolower((unsigned char)name[i]);
    }
    lowered[len] = '\0';
    for (size_t i = 0; i < sizeof(tiers) / sizeof(tiers[0]); ++i) {
        if (strstr(lowered, tiers[i].needle)) {
            free(lowered);
            return tiers[i].tier;
        }
    }
    free(lowered);
    return 20;
}

int ezl_detect_local_capabilities(ezl_string_list *out) {
    if (!out) return EINVAL;
    *out = {};
    try {
        for (const auto &capability : ezlocalai::local_capabilities()) {
            const int result = list_append(out, capability.c_str());
            if (result) { ezl_string_list_free(out); return result; }
        }
        return 0;
    } catch (const std::bad_alloc &) {
        ezl_string_list_free(out);
        return ENOMEM;
    }
}
