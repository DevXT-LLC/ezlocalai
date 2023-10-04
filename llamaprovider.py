import requests


# Instructions for setting up llama.cpp server:
# https://github.com/ggerganov/llama.cpp/tree/master/examples/server#llamacppexampleserver
class LlamaProvider:
    def __init__(
        self,
        STOP_SEQUENCE: str = "</s>",
        MAX_TOKENS: int = 8192,
        AI_TEMPERATURE: float = 0.8,
        AI_TOP_P: float = 0.95,
        AI_TOP_K: int = 40,
        TFS_Z: float = 1.0,
        TYPICAL_P: float = 1.0,
        REPEAT_PENALTY: float = 1.1,
        REPEAT_LAST_N: int = 64,
        PENALIZE_NL: bool = True,
        PRESENCE_PENALTY: float = 0.0,
        FREQUENCY_PENALTY: float = 0.0,
        MIROSTAT: int = 0,
        MIROSTAT_TAU: float = 5.0,
        MIROSTAT_ETA: float = 0.1,
        IGNORE_EOS: bool = False,
        LOGIT_BIAS: list = [],
        **kwargs,
    ):
        self.params = {}
        self.MAX_TOKENS = MAX_TOKENS if MAX_TOKENS else 8192
        if STOP_SEQUENCE:
            self.params["stop"] = STOP_SEQUENCE
        if AI_TEMPERATURE:
            self.params["temperature"] = AI_TEMPERATURE
        if AI_TOP_P:
            self.params["top_p"] = AI_TOP_P
        if AI_TOP_K:
            self.params["top_k"] = AI_TOP_K
        if TFS_Z:
            self.params["tfs_z"] = TFS_Z
        if TYPICAL_P:
            self.params["typical_p"] = TYPICAL_P
        if REPEAT_PENALTY:
            self.params["repeat_penalty"] = REPEAT_PENALTY
        if REPEAT_LAST_N:
            self.params["repeat_last_n"] = REPEAT_LAST_N
        if PENALIZE_NL:
            self.params["penalize_nl"] = PENALIZE_NL
        if PRESENCE_PENALTY:
            self.params["presence_penalty"] = PRESENCE_PENALTY
        if FREQUENCY_PENALTY:
            self.params["frequency_penalty"] = FREQUENCY_PENALTY
        if MIROSTAT:
            self.params["mirostat"] = MIROSTAT
        if MIROSTAT_TAU:
            self.params["mirostat_tau"] = MIROSTAT_TAU
        if MIROSTAT_ETA:
            self.params["mirostat_eta"] = MIROSTAT_ETA
        if IGNORE_EOS:
            self.params["ignore_eos"] = IGNORE_EOS
        if LOGIT_BIAS:
            self.params["logit_bias"] = LOGIT_BIAS

    def instruct(self, prompt, tokens: int = 0):
        self.params["prompt"] = prompt
        self.params["n_predict"] = int(self.MAX_TOKENS) - tokens
        self.params["stream"] = False
        response = requests.post(f"http://localhost:8080/completion", json=self.params)
        data = response.json()
        print(data)
        if "choices" in data:
            choices = data["choices"]
            if choices:
                return choices[0]["text"]
        if "content" in data:
            return data["content"]
        return data
