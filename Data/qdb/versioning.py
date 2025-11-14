"""
数据版本与实验追踪模块 (Versioning)

支持实验版本控制：
实验版本    数据版本    特征版本
RL_v3      qdb_2024Q1  features_v7

使用元数据文件记录版本信息，支持Git LFS / DVC / MLflow集成
"""

import json
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class VersionMetadata:
    """版本元数据"""
    version_id: str
    symbol: str
    data_version: str  # 数据版本（如 qdb_2024Q1）
    feature_version: str = "default"  # 特征版本（如 features_v7）
    experiment_id: str = ""  # 实验ID（如 RL_v3）
    created_at: datetime = None
    description: str = ""
    tags: List[str] = None
    checksum: str = ""  # 数据校验和
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> dict:
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'VersionMetadata':
        d = d.copy()
        d['created_at'] = datetime.fromisoformat(d['created_at'])
        return cls(**d)


class DataVersioning:
    """
    数据版本管理器
    
    功能：
    1. 记录数据版本和实验版本
    2. 支持版本查询和回滚
    3. 生成数据校验和，确保一致性
    4. 支持标签和描述
    """
    
    def __init__(self, base_path: str = "./Data/datasets/qdb"):
        """
        Args:
            base_path: 数据存储基础路径
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.versions_file = self.base_path / "versions.json"
        self._versions: Dict[str, VersionMetadata] = {}
        
        # 加载现有版本
        self._load_versions()
    
    def _load_versions(self):
        """加载版本信息"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    versions_dict = json.load(f)
                    for version_id, meta_dict in versions_dict.items():
                        self._versions[version_id] = VersionMetadata.from_dict(meta_dict)
                logger.info(f"Loaded {len(self._versions)} versions")
            except Exception as e:
                logger.warning(f"Failed to load versions: {e}")
                self._versions = {}
        else:
            self._versions = {}
    
    def _save_versions(self):
        """保存版本信息"""
        versions_dict = {
            version_id: meta.to_dict()
            for version_id, meta in self._versions.items()
        }
        with open(self.versions_file, 'w') as f:
            json.dump(versions_dict, f, indent=2)
        logger.debug(f"Saved {len(self._versions)} versions")
    
    def _compute_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def create_version(self,
                      symbol: str,
                      data_version: str,
                      feature_version: str = "default",
                      experiment_id: str = "",
                      description: str = "",
                      tags: List[str] = None,
                      file_path: Optional[Path] = None) -> str:
        """
        创建新版本
        
        Args:
            symbol: 交易标的
            data_version: 数据版本（如 qdb_2024Q1）
            feature_version: 特征版本（如 features_v7）
            experiment_id: 实验ID（如 RL_v3）
            description: 版本描述
            tags: 标签列表
            file_path: 数据文件路径（用于计算校验和）
        
        Returns:
            版本ID
        """
        # 生成版本ID
        version_id = f"{symbol}_{data_version}_{feature_version}"
        if experiment_id:
            version_id = f"{experiment_id}_{version_id}"
        
        # 计算校验和
        checksum = ""
        if file_path and file_path.exists():
            checksum = self._compute_checksum(file_path)
        
        # 创建版本元数据
        metadata = VersionMetadata(
            version_id=version_id,
            symbol=symbol,
            data_version=data_version,
            feature_version=feature_version,
            experiment_id=experiment_id,
            description=description,
            tags=tags or [],
            checksum=checksum,
        )
        
        # 保存
        self._versions[version_id] = metadata
        self._save_versions()
        
        logger.info(f"Created version: {version_id}")
        return version_id
    
    def get_version(self, version_id: str) -> Optional[VersionMetadata]:
        """获取版本信息"""
        return self._versions.get(version_id)
    
    def list_versions(self,
                     symbol: Optional[str] = None,
                     experiment_id: Optional[str] = None,
                     data_version: Optional[str] = None) -> List[VersionMetadata]:
        """
        列出版本
        
        Args:
            symbol: 筛选symbol
            experiment_id: 筛选实验ID
            data_version: 筛选数据版本
        
        Returns:
            版本列表
        """
        versions = list(self._versions.values())
        
        if symbol:
            versions = [v for v in versions if v.symbol == symbol]
        if experiment_id:
            versions = [v for v in versions if v.experiment_id == experiment_id]
        if data_version:
            versions = [v for v in versions if v.data_version == data_version]
        
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
    
    def get_latest_version(self,
                          symbol: str,
                          experiment_id: Optional[str] = None) -> Optional[VersionMetadata]:
        """获取最新版本"""
        versions = self.list_versions(symbol=symbol, experiment_id=experiment_id)
        if len(versions) > 0:
            return versions[0]
        return None
    
    def add_tag(self, version_id: str, tag: str):
        """添加标签"""
        if version_id in self._versions:
            if tag not in self._versions[version_id].tags:
                self._versions[version_id].tags.append(tag)
                self._save_versions()
    
    def remove_tag(self, version_id: str, tag: str):
        """删除标签"""
        if version_id in self._versions:
            if tag in self._versions[version_id].tags:
                self._versions[version_id].tags.remove(tag)
                self._save_versions()
    
    def verify_checksum(self, version_id: str, file_path: Path) -> bool:
        """验证文件校验和"""
        if version_id not in self._versions:
            return False
        
        metadata = self._versions[version_id]
        if not metadata.checksum:
            logger.warning(f"No checksum stored for version {version_id}")
            return False
        
        current_checksum = self._compute_checksum(file_path)
        return current_checksum == metadata.checksum
    
    def delete_version(self, version_id: str):
        """删除版本"""
        if version_id in self._versions:
            del self._versions[version_id]
            self._save_versions()
            logger.info(f"Deleted version: {version_id}")
    
    def export_for_mlflow(self, version_id: str) -> Dict:
        """
        导出为MLflow格式
        
        Returns:
            MLflow元数据字典
        """
        if version_id not in self._versions:
            return {}
        
        metadata = self._versions[version_id]
        return {
            'version_id': metadata.version_id,
            'symbol': metadata.symbol,
            'data_version': metadata.data_version,
            'feature_version': metadata.feature_version,
            'experiment_id': metadata.experiment_id,
            'description': metadata.description,
            'tags': metadata.tags,
            'created_at': metadata.created_at.isoformat(),
            'checksum': metadata.checksum,
        }















