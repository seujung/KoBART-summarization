import os
import gdown

if os.path.exists('kobart_summary/config.json') and os.path.exists('kobart_summary/pytorch_model.bin') :
    print("== Data existed.==")
    pass
else:
    os.system("rm -rf kobary_summary")
    os.system("mkdir kobart_summary")
    url = "https://drive.google.com/uc?id=1H13loH6dS_2c2Z21kaBtgz42QsjkdAwO"
    output = './kobart_summary/config.json'
    print("Download config.json")
    gdown.download(url, output, quiet=False)

    url = "https://drive.google.com/uc?id=1D7BAXK_0faWW39c0ptE3FtROsVRbTNwI"
    output = './kobart_summary/pytorch_model.bin'
    print("Download pytorch_model.bin")
    gdown.download(url, output, quiet=False)
