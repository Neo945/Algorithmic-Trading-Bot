from flask import Flask, render_template, request, jsonify

from models import bb, random_forest, sma

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get/image')
def get_image():
    stock = request.args.get('stock')
    bar = int(request.args.get('bar'))
    STOCK_DARA = {
        "NIFTY_50": "NSE",
        "AAPL": "NASDAQ",
        "QCOM": "NASDAQ",
        "TCS": "NSE",
        "EBAY": "NASDAQ",
        "CARTRADE": "NSE",
        "NYKAA": "NSE",
        "TATAELXSI": "NSE",
      }
    if request.args.get('strategy') == "SMA":
        sma_p, flag = sma(stock, STOCK_DARA[stock],bar)
        return jsonify({"image": sma_p, "flag": flag})
    elif request.args.get('strategy') == "BBAND":
        bb_p, flag = bb(stock, STOCK_DARA[stock],bar)
        return jsonify({"image": bb_p, "flag": flag})
    else:
        rf, flag = random_forest(stock, STOCK_DARA[stock],bar)
        return jsonify({"image": rf, "flag": flag})

def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()