/* Check the actual installed libraries and the pinned font dependency bundle.
 * gcc tests/brotli_security_smoke.c $(pkg-config --cflags freetype2) \
 *   /opt/ezlocalai/storyboard/lib/libfreetype-*.so.* \
 *   -lbrotlidec -lbrotlienc -ldl -o /tmp/brotli-security-smoke
 * LD_LIBRARY_PATH=/opt/ezlocalai/storyboard/lib /tmp/brotli-security-smoke
 * Pass a different storyboard directory as argv[1] for isolated image checks.
 */
#include <brotli/decode.h>
#include <brotli/encode.h>
#include <dirent.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <ft2build.h>
#include FT_FREETYPE_H

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "/opt/ezlocalai/storyboard";
    const uint32_t required = 0x1002001;
    if (BrotliDecoderVersion() != required || BrotliEncoderVersion() != required)
        return 1;
    char directory[1024], path[2048];
    snprintf(directory, sizeof directory, "%s/lib", root);
    DIR *dir = opendir(directory);
    if (!dir) return 2;
    struct dirent *entry;
    int checked = 0;
    while ((entry = readdir(dir))) {
        if (strncmp(entry->d_name, "libbrotlidec-", sizeof("libbrotlidec-") - 1))
            continue;
        snprintf(path, sizeof path, "%s/%s", directory, entry->d_name);
        void *lib = dlopen(path, RTLD_NOW);
        if (!lib) { fputs(dlerror(), stderr); return 3; }
        uint32_t (*version)(void) = dlsym(lib, "BrotliDecoderVersion");
        if (!version || version() != required) return 4;
        dlclose(lib);
        checked++;
    }
    closedir(dir);
    if (checked != 1) return 5;
    FT_Library ft;
    FT_Face face;
    snprintf(path, sizeof path, "%s/fonts/DejaVuSans.ttf", root);
    if (FT_Init_FreeType(&ft) || FT_New_Face(ft, path, 0, &face)) return 6;
    if (FT_Set_Pixel_Sizes(face, 0, 32) || FT_Load_Char(face, 'A', FT_LOAD_RENDER))
        return 7;
    if (!face->glyph->bitmap.width || !face->glyph->bitmap.rows) return 8;
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
    puts("System and bundled Brotli 1.2.1 checks and FreeType rendering passed");
    return 0;
}
