#include <string>
namespace {
int initialize_error = 0, terminate_error = 0;
std::string error_text, events;
void event(const char *name) { if (!events.empty()) events += ','; events += name; }
}
extern "C" {
void portaudio_fixture_configure(int initialize,int terminate,const char *text) {
    initialize_error = initialize; terminate_error = terminate; error_text = text; events.clear();
}
const char *portaudio_fixture_events() { return events.c_str(); }
int Pa_Initialize() { event("initialize"); return initialize_error; }
int Pa_Terminate() { event("terminate"); return terminate_error; }
const char *Pa_GetErrorText(int) { event("error_text"); return error_text.c_str(); }
}
