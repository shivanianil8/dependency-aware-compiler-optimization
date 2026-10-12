"""
backend/app.py - Flask API for Dependency-Aware Compiler Optimization

This module exposes REST API endpoints to:
1. Check service health: GET /api/health
2. Analyze code, detect applicable passes, and construct the ODG: POST /api/analyze
3. Execute scheduled optimization passes and return transformation diffs: POST /api/optimize

All business logic delegates to:
- backend.optimizer: Parsing, AST generation, semantic validation, and pass transformations.
- backend.dependency_graph: ODG construction, topological sorting, and Cytoscape serialization.
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backend.app")

# Ensure both the backend directory and the project root are on sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import optimizer and dependency_graph modules
try:
    from backend.optimizer import (
        parse_program,
        detect_applicable_passes,
        run_optimization_passes,
        CompilerSyntaxError,
        CompilerSemanticError,
        OptimizationError,
        OptimizerError,
    )
    from backend.dependency_graph import (
        get_optimization_plan,
        build_dependency_graph,
        get_pass_order,
        serialize_graph,
        CyclicDependencyError,
        InvalidPassError,
        DependencyGraphError,
    )
except ImportError:
    from optimizer import (
        parse_program,
        detect_applicable_passes,
        run_optimization_passes,
        CompilerSyntaxError,
        CompilerSemanticError,
        OptimizationError,
        OptimizerError,
    )
    from dependency_graph import (
        get_optimization_plan,
        build_dependency_graph,
        get_pass_order,
        serialize_graph,
        CyclicDependencyError,
        InvalidPassError,
        DependencyGraphError,
    )


# ============================================================================
# Flask Application Setup
# ============================================================================

def create_app() -> Flask:
    """Creates and configures the Flask application."""
    frontend_folder = os.path.join(project_root, "frontend")
    app = Flask(__name__, static_folder=frontend_folder, static_url_path="")
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.route("/")
    def serve_index():
        return app.send_static_file("index.html")

    return app


app = create_app()


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_and_validate_code() -> Tuple[str, Optional[Response]]:
    """
    Validates that the incoming request is JSON and contains a valid 'code' string.
    Returns (code_string, None) on success or ("", error_response) on failure.
    """
    if not request.is_json:
        return "", (
            jsonify({
                "status": "error",
                "error_type": "InvalidInputError",
                "error": "Request content type must be application/json.",
            }),
            400,
        )

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return "", (
            jsonify({
                "status": "error",
                "error_type": "InvalidInputError",
                "error": "Request body must be a valid JSON object.",
            }),
            400,
        )

    if "code" not in data:
        return "", (
            jsonify({
                "status": "error",
                "error_type": "InvalidInputError",
                "error": "Missing required field 'code' in JSON body.",
            }),
            400,
        )

    code = data["code"]
    if not isinstance(code, str):
        return "", (
            jsonify({
                "status": "error",
                "error_type": "InvalidInputError",
                "error": "Field 'code' must be a string.",
            }),
            400,
        )

    return code, None


# ============================================================================
# API Routes
# ============================================================================

@app.route("/api/health", methods=["GET"])
def health_check() -> Tuple[Response, int]:
    """
    GET /api/health
    Returns health status of the backend API.
    """
    return jsonify({
        "status": "healthy",
        "service": "dependency-aware-optimizer-backend",
        "version": "1.0.0",
    }), 200


@app.route("/api/analyze", methods=["POST"])
def analyze_code() -> Tuple[Response, int]:
    """
    POST /api/analyze
    Expects: {"code": "<program>"}

    1. Validates input code syntax and semantics using optimizer.parse_program.
    2. Identifies applicable optimization passes using optimizer.detect_applicable_passes.
    3. Builds the Optimization Dependence Graph (ODG) and computes pass order using
       dependency_graph.get_optimization_plan.
    4. Returns applicable passes, scheduled order, dependencies, and Cytoscape.js graph.
    """
    code, err_resp = _extract_and_validate_code()
    if err_resp is not None:
        return err_resp

    stripped_code = code.strip()
    if not stripped_code:
        # Empty input is handled safely without crashing
        empty_plan = get_optimization_plan([])
        return jsonify({
            "status": "success",
            "code": code,
            "applicable_passes": [],
            "pass_order": [],
            "pass_order_ids": [],
            "dependencies": [],
            "graph": empty_plan["graph"],
            "node_count": 0,
            "edge_count": 0,
            "is_acyclic": True,
            "message": "Empty program provided. No passes are applicable.",
        }), 200

    try:
        # Validate syntax and semantics
        program = parse_program(code, validate=True)

        # Detect which optimization passes can perform work
        applicable_passes = detect_applicable_passes(program)

        # Build dependency graph and determine pass order
        plan = get_optimization_plan(applicable_passes)

        return jsonify({
            "status": "success",
            "code": code,
            "applicable_passes": plan["applicable_passes"],
            "pass_order": plan["pass_order"],
            "pass_order_ids": plan["pass_order_ids"],
            "dependencies": plan["dependencies"],
            "graph": plan["graph"],
            "node_count": plan["node_count"],
            "edge_count": plan["edge_count"],
            "is_acyclic": plan["is_acyclic"],
        }), 200

    except CompilerSyntaxError as e:
        logger.warning("Syntax error in /api/analyze: %s", str(e))
        return jsonify({
            "status": "error",
            "error_type": "SyntaxError",
            "error": str(e),
        }), 400

    except CompilerSemanticError as e:
        logger.warning("Semantic error in /api/analyze: %s", str(e))
        return jsonify({
            "status": "error",
            "error_type": "SemanticError",
            "error": str(e),
        }), 400

    except CyclicDependencyError as e:
        logger.error("Cyclic dependency error in /api/analyze: %s", str(e))
        return jsonify({
            "status": "error",
            "error_type": "CyclicDependencyError",
            "error": str(e),
        }), 400

    except (InvalidPassError, ValueError) as e:
        logger.error("Pass error in /api/analyze: %s", str(e))
        return jsonify({
            "status": "error",
            "error_type": "InvalidPassError",
            "error": str(e),
        }), 400

    except Exception as e:
        logger.exception("Unexpected error in /api/analyze")
        return jsonify({
            "status": "error",
            "error_type": "InternalServerError",
            "error": "An unexpected error occurred while analyzing the code.",
        }), 500


@app.route("/api/optimize", methods=["POST"])
def optimize_code() -> Tuple[Response, int]:
    """
    POST /api/optimize
    Expects: {"code": "<program>", "pass_order": ["..."] (optional)}

    1. Validates and parses the program.
    2. Builds the ODG and calculates the deterministic topological pass order.
    3. Executes the optimization passes in order using optimizer.run_optimization_passes.
    4. Returns original code, optimized code, transformation log, and serialized ODG.
    """
    code, err_resp = _extract_and_validate_code()
    if err_resp is not None:
        return err_resp

    data = request.get_json(silent=True) or {}
    custom_pass_order = data.get("pass_order", None)

    stripped_code = code.strip()
    if not stripped_code:
        empty_plan = get_optimization_plan([])
        return jsonify({
            "status": "success",
            "original_code": code,
            "optimized_code": code,
            "applicable_passes": [],
            "selected_pass_order": [],
            "transformation_log": [],
            "stats": {
                "transformations_count": 0,
                "original_statement_count": 0,
                "optimized_statement_count": 0,
            },
            "graph": empty_plan["graph"],
            "dependencies": [],
            "message": "Empty program provided. No optimizations performed.",
        }), 200

    try:
        # Validate syntax and semantics
        program = parse_program(code, validate=True)

        # Detect applicable passes
        applicable_passes = detect_applicable_passes(program)

        # Build dependency graph and determine pass order
        plan = get_optimization_plan(applicable_passes)

        # Use custom pass order if explicitly supplied and valid, otherwise use ODG order
        if custom_pass_order is not None:
            if not isinstance(custom_pass_order, list):
                return jsonify({
                    "status": "error",
                    "error_type": "InvalidInputError",
                    "error": "Field 'pass_order' must be a list of pass names.",
                }), 400
            execution_order = custom_pass_order
        else:
            execution_order = plan["pass_order"]

        # Execute optimization pipeline through optimizer module
        opt_result = run_optimization_passes(program, execution_order)

        return jsonify({
            "status": "success",
            "original_code": opt_result["original_code"],
            "optimized_code": opt_result["optimized_code"],
            "applicable_passes": plan["applicable_passes"],
            "selected_pass_order": execution_order,
            "transformation_log": opt_result["transformation_log"],
            "stats": opt_result["stats"],
            "graph": plan["graph"],
            "dependencies": plan["dependencies"],
        }), 200

    except CompilerSyntaxError as e:
        logger.warning("Syntax error in /api/optimize: %s", str(e))
        return jsonify({
            "status": "error",
            "error_type": "SyntaxError",
            "error": str(e),
        }), 400

    except CompilerSemanticError as e:
        logger.warning("Semantic error in /api/optimize: %s", str(e))
        return jsonify({
            "status": "error",
            "error_type": "SemanticError",
            "error": str(e),
        }), 400

    except OptimizationError as e:
        logger.warning("Optimization error in /api/optimize: %s", str(e))
        return jsonify({
            "status": "error",
            "error_type": "OptimizationError",
            "error": str(e),
        }), 400

    except CyclicDependencyError as e:
        logger.error("Cyclic dependency error in /api/optimize: %s", str(e))
        return jsonify({
            "status": "error",
            "error_type": "CyclicDependencyError",
            "error": str(e),
        }), 400

    except (InvalidPassError, ValueError) as e:
        logger.error("Pass error in /api/optimize: %s", str(e))
        return jsonify({
            "status": "error",
            "error_type": "InvalidPassError",
            "error": str(e),
        }), 400

    except Exception as e:
        logger.exception("Unexpected error in /api/optimize")
        return jsonify({
            "status": "error",
            "error_type": "InternalServerError",
            "error": "An unexpected error occurred while optimizing the code.",
        }), 500


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    def get_port() -> int:
        env_port = os.environ.get("PORT")
        if env_port:
            return int(env_port)
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 5000))
            s.close()
            return 5000
        except OSError:
            logger.info("Port 5000 is occupied (e.g. macOS AirPlay Receiver). Using port 5001.")
            return 5001

    port = get_port()
    logger.info("Starting Dependency-Aware Compiler Optimization server on http://127.0.0.1:%d", port)
    app.run(host="127.0.0.1", port=port, debug=True)
