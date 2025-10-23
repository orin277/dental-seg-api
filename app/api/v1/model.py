from fastapi import UploadFile, File, Request, Depends, status, APIRouter
from fastapi.responses import JSONResponse


from app.services.model_service import ModelService
from app.dependencies.model import get_model_service


router = APIRouter(
    prefix="/predict",
    tags=["Model"]
)

@router.post('/tooth-segmentation')
async def predict_teeth(
    request: Request, 
    file: UploadFile = File(...),
    model_service: ModelService = Depends(get_model_service)
) -> JSONResponse:
    path = await model_service.predict_teeth(file, request.app.state.tooth_seg_models)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content = {
            "mask_path": path
        }
    )


@router.post('/caries-segmentation')
async def predict_caries(
    request: Request, 
    file: UploadFile = File(...),
    model_service: ModelService = Depends(get_model_service)
) -> JSONResponse:
    path = await model_service.predict_caries(
        file, 
        request.app.state.tooth_seg_models, 
        request.app.state.caries_seg_models
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content = {
            "mask_path": path
        }
    )