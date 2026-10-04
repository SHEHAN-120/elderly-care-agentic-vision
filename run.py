"""
Convenience launcher.
  python run.py                        → starts the web server (frontend + API)
  python run.py analyze <video> [opts] → runs CLI analysis
"""
import sys
import os


def start_server():
    import uvicorn
    print("\n🏥 Elderly Activity Monitoring — starting server...")
    print("   Open:  http://localhost:8000\n")
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=8000, reload=False)


def run_cli():
    from backend.main import main
    sys.argv = ["backend.main"] + sys.argv[2:]
    main()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "analyze":
        run_cli()
    else:
        start_server()