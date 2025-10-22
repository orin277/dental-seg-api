from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MASK_PATH: str
    DEVICE: str

    TOOTH_SEG_MODEL1_PATH: str
    TOOTH_SEG_MODEL2_PATH: str
    TOOTH_SEG_MODEL3_PATH: str

    CARIES_SEG_MODEL1_PATH: str
    CARIES_SEG_MODEL2_PATH: str
    CARIES_SEG_MODEL3_PATH: str

    TOOTH_IMG_SIZE: int
    CARIES_IMG_SIZE: int

    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8'
    )

    @property
    def all_tooth_seg_model_paths(self):
        return [
            self.TOOTH_SEG_MODEL1_PATH, 
            self.TOOTH_SEG_MODEL2_PATH, 
            self.TOOTH_SEG_MODEL3_PATH
        ]
    
    @property
    def all_caries_seg_model_paths(self):
        return [
            self.CARIES_SEG_MODEL1_PATH, 
            self.CARIES_SEG_MODEL2_PATH, 
            self.CARIES_SEG_MODEL3_PATH
        ]


settings = Settings()