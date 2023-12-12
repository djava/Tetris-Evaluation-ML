import socketserver
import json
import typing
from http.server import BaseHTTPRequestHandler
import constants
from models.ModelBase import ModelBase
from utils import generate_ptp_terms

_models: dict = {}


def predict_eval(heights: list[int]) -> dict[str, float]:
    df = generate_ptp_terms(heights)
    return {str(mt): mb.predict(df) for mt, mb in _models.items()}


def get_model_test_mses() -> dict[str, float]:
    return {str(mt): mb.test_mse for mt, mb in _models.items()}


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        print(f"Received GET request for {self.path}")

        content_length = int(self.headers['Content-Length'])
        request_body = self.rfile.read(content_length).decode('utf-8')
        if self.path == '/model-info':
            self.handle_model_info()
        elif self.path == '/predict':
            self.handle_predict(request_body)
        elif self.path == '/multi-predict':
            self.handle_multi_predict(request_body)
        else:
            self.handle_404()

    def handle_predict(self, post_data: str) -> None:
        try:
            json_data = json.loads(post_data)
            eval_prediction = predict_eval(json_data['heights'])
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
            json_data = json.loads(post_data)
            eval_predictions = {
                "eval": [predict_eval(heights) for heights in json_data['heights'] if is_valid(heights)]
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
    with socketserver.TCPServer(("", constants.SERVER_PORT), RequestHandler) as httpd:
        print(f"Serving on port: {constants.SERVER_PORT}")
        httpd.serve_forever()
