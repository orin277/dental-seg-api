from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import settings
from app.neural_networks.models.model_loader import ModelLoader
from app.api.v1.model import router as model_router
from app.middlewares.process_time_header_middleware import ProcessTimeHeaderMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tooth_seg_models = ModelLoader.load_tooth_ensemble(
        settings.all_tooth_seg_model_paths,
        settings.DEVICE
    )
    app.state.caries_seg_models = ModelLoader.load_caries_ensemble(
        settings.all_caries_seg_model_paths,
        settings.DEVICE
    )

    yield


app = FastAPI(lifespan=lifespan)

app.include_router(model_router)

app.add_middleware(ProcessTimeHeaderMiddleware)