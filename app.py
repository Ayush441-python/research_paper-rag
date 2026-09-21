from fastapi import FastAPI


from apis import health
from apis import upload


app = FastAPI()


app.include_router(health.router)
app.include_router(upload.router)




if __name__ =="__main__":
    import uvicorn
    uvicorn.run(app)