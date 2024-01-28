from local_llm.LLM import LLM
from local_llm.STT import STT
from local_llm.CTTS import CTTS


def display_content(content, outputs_url="http://localhost:8091/outputs/"):
    try:
        from IPython.display import Audio, display, Image, Video
    except:
        print(content)
        return
    if "<audio controls>" in content:
        import base64
        from datetime import datetime

        audio_response = content.split("data:audio/wav;base64,")[1].split('" type')[0]
        file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.wav"
        with open(file_name, "wb") as fh:
            fh.write(base64.b64decode(audio_response))
        display(Audio(filename=file_name, autoplay=True))
    if outputs_url in content:
        file_name = content.split(outputs_url)[1].split('"')[0]
        url = f"{outputs_url}{file_name}"
        if url.endswith(".jpg") or url.endswith(".png"):
            content = content.replace(url, "")
            display(Image(url=url))
        elif url.endswith(".mp4"):
            content = content.replace(url, "")
            display(Video(url=url, autoplay=True))
        elif url.endswith(".wav"):
            content = content.replace(url, "")
            display(Audio(url=url, autoplay=True))
    print(content)
