import uvicorn

if __name__ == "__main__":
    """
    This is the main entry point for the application.
    It uses uvicorn to run the FastAPI application defined in 'app/main.py'.
    """
    uvicorn.run(
        "app.main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )
