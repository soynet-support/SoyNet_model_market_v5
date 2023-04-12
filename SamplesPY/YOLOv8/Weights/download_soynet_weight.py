import wget
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--path", help="download path", type=str, default='../../../mgmt/weights')
    args = parser.parse_args()

    weight_url = [
        'https://kr.object.iwinv.kr/model_market_weight_v5/yolov8s.weights',
        'https://kr.object.iwinv.kr/model_market_weight_v5/yolov8n.weights',
        'https://kr.object.iwinv.kr/model_market_weight_v5/yolov8m.weights',
        'https://kr.object.iwinv.kr/model_market_weight_v5/yolov8l.weights'

    ]

    for url in weight_url:
        wget.download(url, args.path)



if __name__ == '__main__':
    main()
