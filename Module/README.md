# OVCD Enhancement Modules

OVCD增强模块系统为开放词汇变化检测提供即插即用的改进组件。

## 🎯 特性

- **模块化设计**: 标准化的模块接口，易于扩展
- **即插即用**: 无需修改核心代码，通过配置文件启用
- **轻量级**: 最小化计算开销和内存占用
- **类型化**: 支持多种类型的增强模块（阈值调整、特征增强、后处理等）

## 📁 项目结构

```
Module/
├── __init__.py                    # 模块包初始化
├── base.py                        # 基础抽象类
├── preprocessing_base.py          # 预处理模块基础类
├── registry.py                    # 模块注册系统
├── config_manager.py              # 配置管理器
├── integrator.py                  # 管道集成器
├── otsu_adaptive_threshold.py     # Otsu自适应阈值模块
├── acdp_preprocessing.py          # ACDP Advanced Change Detection Preprocessor
├── configs/                       # 配置文件目录
│   ├── otsu_adaptive_threshold_example.yaml
│   └── acdp_preprocessing_config.yaml
└── README.md                      # 本文档
```

## 🚀 快速开始

### 1. 基础使用

```python
from Module import get_module

# 创建Otsu自适应阈值模块
config = {
    'global_weight': 0.6,
    'edge_weight': 0.4,
    'debug': True
}
module = get_module('otsu_adaptive_threshold', config)
```

### 2. 配置文件使用

创建模块配置文件：

```yaml
# configs/my_modules.yaml
module_name: "otsu_adaptive_threshold"
module_type: "threshold_adjustment"
enabled: true
config:
  global_weight: 0.6
  edge_weight: 0.4
  edge_detection_method: "canny"
  debug: true
```

在OVCD配置中启用：

```yaml
# configs/models/sam_dinov2_dgtrs_modular.yaml
comparator:
  type: "DINOv2"
  enable_otsu_adaptive: true
  otsu_adaptive_config:
    global_weight: 0.6
    edge_weight: 0.4
    debug: true
```

### 3. 运行测试

```bash
cd OVCD
python test_modular_system.py
```

## 📦 可用模块

### Otsu自适应阈值模块

**模块名**: `otsu_adaptive_threshold`  
**类型**: `threshold_adjustment`  
**功能**: 基于全局+局部Otsu阈值的自适应阈值调整

**配置参数**:
- `global_weight` (float): 全局Otsu权重 (0.0-1.0)
- `edge_weight` (float): 边缘Otsu权重 (0.0-1.0)  
- `edge_detection_method` (str): 边缘检测方法 ("canny" 或 "sobel")
- `canny_low` (int): Canny低阈值
- `canny_high` (int): Canny高阈值
- `edge_dilation_kernel` (int): 边缘膨胀核大小
- `min_edge_pixels` (int): 边缘Otsu最小像素数
- `debug` (bool): 调试输出开关

### ACDP预处理增强模块

**模块名**: `acdp_preprocessing`  
**类型**: `preprocessing_enhancement`  
**功能**: Advanced Change Detection Preprocessor - 高效双时相图像增强框架

**核心特性**:
- **轻量级设计**: 基于成熟的图像处理技术，计算效率高
- **时序对齐**: 智能光照归一化，减少季节和天气影响
- **数据增强**: 同步双时相增强，保持空间对应关系
- **自适应处理**: 根据图像特征自动调整处理参数

**主要配置参数**:

*核心设置*:
- `enable_augmentation` (bool): 是否启用数据增强
- `img_size` (int/null): 目标图像尺寸，null为动态尺寸
- `normalize_method` (str): 归一化方法 ("standard", "minmax", null)
- `temporal_alignment` (bool): 启用双时相光照一致性对齐

*增强控制*:
- `with_random_hflip` (bool): 随机水平翻转
- `with_random_vflip` (bool): 随机垂直翻转
- `with_scale_random_crop` (bool): 尺度感知随机裁剪
- `with_color_jitter` (bool): 随机颜色变换
- `adaptive_contrast` (bool): 自适应对比度增强

**使用示例**:
```yaml
segmentor:
  type: "SAM"
  enable_acdp_preprocessing: true              # 🔥 主开关
  acdp_preprocessing_config:
    # 核心设置
    enable_augmentation: false                # 推理模式关闭增强
    normalize_method: "standard"              # 标准归一化
    temporal_alignment: true                  # 光照一致性对齐
    
    # 高级设置
    adaptive_contrast: false                  # 保持稳定性能
    verbose_logging: false                    # 简洁输出
```

**性能模式**:
- **生产模式**: `enable_augmentation: false`, `temporal_alignment: true`
- **训练模式**: `enable_augmentation: true`, 所有增强启用
- **快速模式**: `temporal_alignment: false`, 最小处理

## 🔧 开发自定义模块

### 1. 创建模块类

```python
from Module.base import ThresholdAdjustmentModule

class MyCustomModule(ThresholdAdjustmentModule):
    def __init__(self, config):
        super().__init__(config)
        # 初始化参数
    
    def initialize(self):
        # 模块初始化逻辑
        self.is_initialized = True
    
    def compute_adaptive_threshold(self, img1_embed, img2_embed, base_threshold, **kwargs):
        # 实现自适应阈值计算
        return adjusted_threshold
    
    def get_config_template(self):
        return {
            'param1': 'default_value1',
            'param2': 'default_value2'
        }
```

### 2. 注册模块

```python
from Module import register_module

register_module('my_custom_module', MyCustomModule)
```

### 3. 使用模块

```python
from Module import get_module

module = get_module('my_custom_module', config)
```

## 🔄 集成流程

1. **模块注册**: 模块在导入时自动注册到全局注册表
2. **配置加载**: 通过配置文件或直接传参加载模块配置
3. **模块实例化**: 根据配置创建模块实例
4. **管道集成**: 模块自动集成到OVCD比较器中
5. **运行时调用**: 在变化检测过程中自动调用模块方法

## 📊 性能考虑

- **计算开销**: Otsu阈值计算仅需直方图分析，开销极小
- **内存占用**: 模块设计为轻量级，避免大量内存分配
- **缓存机制**: 支持模块实例缓存，避免重复初始化

## 🐛 调试和测试

### 启用调试输出

```python
config = {
    'debug': True,
    # 其他参数...
}
```

### 运行测试脚本

```bash
python test_modular_system.py
```

### 检查模块状态

```python
from Module import list_available_modules, get_module_info

# 列出所有可用模块
modules = list_available_modules()
print(f"Available modules: {modules}")

# 获取模块详细信息
info = get_module_info('otsu_adaptive_threshold')
print(f"Module info: {info}")
```

## 🔮 未来扩展

计划支持的模块类型：

1. **特征增强模块** (`feature_enhancement`)
   - 特征归一化
   - 特征融合
   - 注意力机制

2. **后处理模块** (`post_processing`)
   - 形态学操作
   - 连通域分析
   - 置信度校准

3. **预处理模块** (`pre_processing`)
   - 图像增强
   - 噪声去除
   - 对比度调整

## 📝 注意事项

1. **向后兼容**: 现有配置文件无需修改即可使用
2. **错误处理**: 模块加载失败时会回退到原始方法
3. **配置验证**: 支持配置参数验证，确保参数有效性
4. **文档生成**: 支持自动生成模块文档

## 🤝 贡献指南

1. 继承合适的基类 (`ThresholdAdjustmentModule`, `FeatureEnhancementModule`, 等)
2. 实现必需的抽象方法
3. 提供完整的配置模板
4. 添加单元测试
5. 更新文档

---

**作者**: OVCD团队  
**版本**: 1.0.0  
**更新时间**: 2024年