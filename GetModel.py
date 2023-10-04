import argparse
from provider import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_url", type=str, default="None")
    parser.add_argument("--quant_type", type=str, default="Q4_K_M")
    args = parser.parse_args()
    model_url = args.model_url
    quant_type = args.quant_type
    if model_url != "None":
        model_path = get_model(model_url, quant_type)
        print(model_path)
