from flask import Flask, jsonify, redirect, request

from systems import Ranker, Recommender

app = Flask(__name__)
ranker = Ranker()
recommender = Recommender()


@app.route("/")
def redirect_to_test():
    return redirect("/test", code=302)


@app.route("/test", methods=["GET"])
def test():
    return "Container is running", 200


@app.route("/index", methods=["GET"])
def index():
    ranker.index()
    recommender.index()
    return "Indexing done!", 200


@app.route("/ranking", methods=["GET"])
def ranking():
    query = request.args.get("query", None)
    page = request.args.get("page", default=0, type=int)
    rpp = request.args.get("rpp", default=20, type=int)
    response = ranker.rank_publications(query, page, rpp)
    return jsonify(response)


@app.route("/recommendation", methods=["GET"])
def rec_combined():
    item_id = request.args.get("item_id", None)
    page = request.args.get("page", default=0, type=int)
    rpp = request.args.get("rpp", default=20, type=int)

    publications = recommender.recommend(item_id, page, rpp)

    return jsonify(publications)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
