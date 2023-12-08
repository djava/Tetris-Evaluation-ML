import socketserver
import json
from http.server import BaseHTTPRequestHandler
import constants

_models: dict = {}


def predict_eval(heights: list[int]) -> dict[str, float]:
    return {str(mt): mb.predict(heights) for mt, mb in _models.items()}


def get_model_test_mses() -> dict[str, float]:
    return {str(mt): mb.test_mse for mt, mb in _models.items()}


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f"Received GET request for {self.path}")
        if self.path == '/model_info':
            self.handle_model_info()
        else:
            self.handle_404()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')

        print(f"Received POST request for {self.path}: {post_data}")
        if self.path == '/predict':
            self.handle_predict(post_data)
        else:
            self.handle_404()

    def handle_predict(self, post_data):
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
            # Handle invalid JSON data
            self.send_response(400)
            self.send_header("Content-type", "application/html")
            self.end_headers()
            self.wfile.write("Invalid JSON data".encode())
            print(f"ERROR: {e}")

    def handle_model_info(self):
        test_mses = get_model_test_mses()
        response_message = json.dumps(test_mses)

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(response_message.encode())
        print(f"Sent response: {response_message}")

    def handle_404(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'Not Found')


def run_server(models):
    global _models
    _models = models

    # Create the server, binding to localhost on the specified port
    with socketserver.TCPServer(("", constants.SERVER_PORT), RequestHandler) as httpd:
        print(f"Serving on port: {constants.SERVER_PORT}")
        httpd.serve_forever()
