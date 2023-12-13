import socketserver
import json
import time
import typing
from http.server import BaseHTTPRequestHandler

from constants import *
from models.ModelBase import ModelBase
from utils import generate_ptp_terms, inverse_normalized_eval
from joblib import Parallel, delayed, cpu_count
from functools import lru_cache

_models: dict[ModelID, ModelBase] = {}


# @lru_cache()
def predict_eval(heights: tuple[int], model: ModelBase) -> float:
    df = generate_ptp_terms(heights)

    prediction = model.predict(df)
    if model.dataset_norm is DataSetNorm.NORMALIZED:
        prediction = inverse_normalized_eval(prediction)

    return prediction


def get_model_test_mses() -> dict[str, float]:
    return {str(mt): mb.test_mse for mt, mb in _models.items()}


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        print(f"Received GET request for {self.path}")
        if self.path == '/model-info':
            self.handle_model_info()
        elif self.path == '/cpu-info':
            self.handle_cpu_info()
        else:
            self.handle_404()

    def do_POST(self) -> None:
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')

        print(f"Received POST request for {self.path}: {post_data}")
        if self.path == '/predict':
            self.handle_predict(post_data)
        elif self.path == '/multi-predict':
            self.handle_multi_predict(post_data)
        else:
            self.handle_404()

    def handle_predict(self, post_data: str) -> None:
        try:
            json_data = json.loads(post_data)
            model = _models[ModelID(json_data['model'])]
            eval_prediction = predict_eval(json_data['heights'], model)
            response_message = json.dumps(eval_prediction)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(response_message.encode())
            print(f"Sent response: {response_message}")
        except json.JSONDecodeError as e:
            self.handle_error(e)

    def handle_multi_predict(self, post_data: str) -> None:
        def is_valid(heights: list[int]) -> bool:
            if not isinstance(heights, list):
                return False
            return len(heights) == 10 and all(isinstance(i, int) for i in heights)

        try:
            start_time = time.monotonic()
            json_data = json.loads(post_data)
            model = _models[ModelID(json_data['model'])]

            if 'multiplier' in json_data.keys():
                json_data['heights'] = json_data['heights'] * json_data['multiplier']

            parallel = Parallel(n_jobs=cpu_count())
            eval_results = list(parallel(delayed(predict_eval)(tuple(h), model)
                                         for h in json_data['heights']
                                         if is_valid(h)))

            eval_predictions = {
                "eval": eval_results,
                "time_elapsed": time.monotonic() - start_time
            }
            response_message = json.dumps(eval_predictions)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(response_message.encode())
            print(f"Sent response: {response_message}")
        except json.JSONDecodeError as e:
            self.handle_error(e)
        except ValueError as e:
            self.handle_error(e)

    def handle_model_info(self) -> None:
        test_mses = get_model_test_mses()
        response_message = json.dumps(test_mses)

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(response_message.encode())
        print(f"Sent response: {response_message}")

    def handle_cpu_info(self) -> None:
        response_message = str(joblib.cpu_count())

        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(response_message.encode())
        print(f"Sent response: {response_message}")

    def handle_404(self) -> None:
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'Not Found')

    def handle_error(self, e: Exception) -> None:
        self.send_response(400)
        self.send_header("Content-type", "application/html")
        self.end_headers()
        self.wfile.write(f"Invalid request:\n{e}".encode())
        print(f"ERROR: {e}")


def run_server(models: list[ModelBase]) -> typing.NoReturn:
    global _models
    _models = models

    # Create the server, binding to localhost on the specified port
    with socketserver.TCPServer(("", SERVER_PORT), RequestHandler) as httpd:
        print(f"Serving on port: {SERVER_PORT}")
        httpd.serve_forever()
