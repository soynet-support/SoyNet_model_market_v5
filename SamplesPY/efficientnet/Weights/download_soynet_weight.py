import wget
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--path", help="download path", type=str, default='../../../mgmt/weights')
    args = parser.parse_args()

    weight_url = [
        'https://kr.object.iwinv.kr/model_market_weight_v5/efficientnet-b0.weights',
        'https://kr.object.iwinv.kr/model_market_weight_v5/efficientnet-b1.weights',
        'https://kr.object.iwinv.kr/model_market_weight_v5/efficientnet-b2.weights',
        'https://kr.object.iwinv.kr/model_market_weight_v5/efficientnet-b3.weights',
        'https://kr.object.iwinv.kr/model_market_weight_v5/efficientnet-b4.weights',
        'https://kr.object.iwinv.kr/model_market_weight_v5/efficientnet-b5.weights',
        'https://kr.object.iwinv.kr/model_market_weight_v5/efficientnet-b6.weights',
        'https://kr.object.iwinv.kr/model_market_weight_v5/efficientnet-b7.weights'

    ]

    for url in weight_url:
        wget.download(url, args.path)



if __name__ == '__main__':
    main()
