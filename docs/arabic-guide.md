# دليل ThinkingNet-Go باللغة العربية

## مقدمة

مرحباً بك في دليل مكتبة ThinkingNet-Go الشامل باللغة العربية! هذه المكتبة هي أداة قوية ومرنة لتعلم الآلة والذكاء الاصطناعي مكتوبة بلغة Go. تم تصميمها لتكون سهلة الاستخدام للمبتدئين وقوية بما فيه الكفاية للخبراء.

## المحتويات

1. [التثبيت والإعداد](#التثبيت-والإعداد)
2. [البداية السريعة](#البداية-السريعة)
3. [الدروس التطبيقية](#الدروس-التطبيقية)
4. [الأنماط الشائعة](#الأنماط-الشائعة)
5. [أفضل الممارسات](#أفضل-الممارسات)
6. [استكشاف الأخطاء وإصلاحها](#استكشاف-الأخطاء-وإصلاحها)
7. [مرجع الواجهات البرمجية](#مرجع-الواجهات-البرمجية)

## التثبيت والإعداد

### المتطلبات الأساسية

- Go 1.19 أو أحدث
- Git للحصول على المكتبة

### خطوات التثبيت

```bash
# إنشاء مشروع جديد
mkdir my-ml-project
cd my-ml-project

# تهيئة وحدة Go
go mod init my-ml-project

# إضافة مكتبة ThinkingNet-Go
go get github.com/blackmoon87/thinkingnet
```

### التحقق من التثبيت

```go
package main

import (
    "fmt"
    "github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
    // إنشاء tensor بسيط للتحقق من عمل المكتبة
    data := [][]float64{{1, 2}, {3, 4}}
    tensor := core.EasyTensor(data)
    fmt.Println("المكتبة تعمل بشكل صحيح!")
    fmt.Println("البيانات:", tensor.Shape())
}
```

## البداية السريعة

### مثال بسيط: تصنيف XOR

```go
package main

import (
    "fmt"
    "github.com/blackmoon87/thinkingnet/pkg/models"
    "github.com/blackmoon87/thinkingnet/pkg/layers"
    "github.com/blackmoon87/thinkingnet/pkg/activations"
    "github.com/blackmoon87/thinkingnet/pkg/optimizers"
    "github.com/blackmoon87/thinkingnet/pkg/losses"
    "github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
    // 1. إعداد البيانات
    X := core.EasyTensor([][]float64{
        {0, 0}, {0, 1}, {1, 0}, {1, 1},
    })
    y := core.EasyTensor([][]float64{
        {0}, {1}, {1}, {0},
    })

    // 2. إنشاء النموذج
    model := models.NewSequential()
    model.AddLayer(layers.NewDense(4, activations.NewReLU()))
    model.AddLayer(layers.NewDense(1, activations.NewSigmoid()))

    // 3. تجميع النموذج
    model.Compile(optimizers.NewAdam(0.01), losses.NewBinaryCrossEntropy())

    // 4. تدريب النموذج (استخدام الدالة المبسطة)
    fmt.Println("بدء التدريب...")
    history, err := model.EasyTrain(X, y)
    if err != nil {
        fmt.Printf("خطأ في التدريب: %v\n", err)
        return
    }

    // 5. التنبؤ
    predictions, err := model.EasyPredict(X)
    if err != nil {
        fmt.Printf("خطأ في التنبؤ: %v\n", err)
        return
    }

    // 6. عرض النتائج
    fmt.Println("التدريب مكتمل!")
    fmt.Printf("آخر خسارة: %.4f\n", history.Loss[len(history.Loss)-1])
    fmt.Println("التنبؤات:", predictions.Data())
}
```

## الدروس التطبيقية

### الدرس الأول: الانحدار الخطي

#### الهدف
تعلم كيفية إنشاء نموذج انحدار خطي بسيط لتوقع الأسعار.

#### الخطوات

```go
package main

import (
    "fmt"
    "github.com/blackmoon87/thinkingnet/pkg/algorithms"
    "github.com/blackmoon87/thinkingnet/pkg/preprocessing"
    "github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
    // 1. إعداد بيانات وهمية (المساحة -> السعر)
    // المساحة بالمتر المربع
    areas := [][]float64{
        {50}, {75}, {100}, {125}, {150}, {175}, {200},
    }
    // السعر بالآلاف
    prices := [][]float64{
        {100}, {150}, {200}, {250}, {300}, {350}, {400},
    }

    X := core.EasyTensor(areas)
    y := core.EasyTensor(prices)

    // 2. تقسيم البيانات
    XTrain, XTest, yTrain, yTest := preprocessing.EasySplit(X, y, 0.3)

    // 3. تطبيع البيانات
    XTrainScaled := preprocessing.EasyStandardScale(XTrain)
    XTestScaled := preprocessing.EasyStandardScale(XTest)

    // 4. إنشاء وتدريب النموذج
    model := algorithms.EasyLinearRegression()
    
    err := model.Fit(XTrainScaled, yTrain)
    if err != nil {
        fmt.Printf("خطأ في التدريب: %v\n", err)
        return
    }

    // 5. التنبؤ والتقييم
    predictions, err := model.Predict(XTestScaled)
    if err != nil {
        fmt.Printf("خطأ في التنبؤ: %v\n", err)
        return
    }

    fmt.Println("نموذج الانحدار الخطي جاهز!")
    fmt.Println("التنبؤات:", predictions.Data())
    fmt.Println("القيم الحقيقية:", yTest.Data())
}
```

### الدرس الثاني: التصنيف باستخدام الانحدار اللوجستي

#### الهدف
تصنيف البيانات إلى فئتين باستخدام الانحدار اللوجستي.

```go
package main

import (
    "fmt"
    "math/rand"
    "github.com/blackmoon87/thinkingnet/pkg/algorithms"
    "github.com/blackmoon87/thinkingnet/pkg/preprocessing"
    "github.com/blackmoon87/thinkingnet/pkg/core"
    "github.com/blackmoon87/thinkingnet/pkg/metrics"
)

func main() {
    // 1. إنشاء بيانات تصنيف وهمية
    var features [][]float64
    var labels [][]float64
    
    rand.Seed(42)
    for i := 0; i < 100; i++ {
        x1 := rand.Float64()*10 - 5  // قيم بين -5 و 5
        x2 := rand.Float64()*10 - 5
        
        // قاعدة بسيطة للتصنيف
        label := 0.0
        if x1 + x2 > 0 {
            label = 1.0
        }
        
        features = append(features, []float64{x1, x2})
        labels = append(labels, []float64{label})
    }

    X := core.EasyTensor(features)
    y := core.EasyTensor(labels)

    // 2. تقسيم البيانات
    XTrain, XTest, yTrain, yTest := preprocessing.EasySplit(X, y, 0.2)

    // 3. تطبيع البيانات
    XTrainScaled := preprocessing.EasyStandardScale(XTrain)
    XTestScaled := preprocessing.EasyStandardScale(XTest)

    // 4. إنشاء وتدريب النموذج
    model := algorithms.EasyLogisticRegression()
    
    err := model.Fit(XTrainScaled, yTrain)
    if err != nil {
        fmt.Printf("خطأ في التدريب: %v\n", err)
        return
    }

    // 5. التنبؤ
    predictions, err := model.Predict(XTestScaled)
    if err != nil {
        fmt.Printf("خطأ في التنبؤ: %v\n", err)
        return
    }

    // 6. تقييم الأداء
    accuracy := metrics.Accuracy(yTest, predictions)
    
    fmt.Println("نموذج التصنيف جاهز!")
    fmt.Printf("دقة النموذج: %.2f%%\n", accuracy*100)
}
```

### الدرس الثالث: التجميع باستخدام K-Means

#### الهدف
تجميع البيانات إلى مجموعات متشابهة.

```go
package main

import (
    "fmt"
    "math/rand"
    "github.com/blackmoon87/thinkingnet/pkg/algorithms"
    "github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
    // 1. إنشاء بيانات تجميع وهمية
    var data [][]float64
    
    rand.Seed(42)
    
    // مجموعة 1: حول النقطة (2, 2)
    for i := 0; i < 30; i++ {
        x := 2 + rand.Float64()*2 - 1  // بين 1 و 3
        y := 2 + rand.Float64()*2 - 1
        data = append(data, []float64{x, y})
    }
    
    // مجموعة 2: حول النقطة (8, 8)
    for i := 0; i < 30; i++ {
        x := 8 + rand.Float64()*2 - 1  // بين 7 و 9
        y := 8 + rand.Float64()*2 - 1
        data = append(data, []float64{x, y})
    }

    X := core.EasyTensor(data)

    // 2. إنشاء نموذج K-Means
    model := algorithms.EasyKMeans(2)  // تجميع إلى مجموعتين

    // 3. تدريب النموذج
    err := model.Fit(X)
    if err != nil {
        fmt.Printf("خطأ في التدريب: %v\n", err)
        return
    }

    // 4. الحصول على التسميات
    labels := model.GetLabels()
    centers := model.GetCenters()

    fmt.Println("تم التجميع بنجاح!")
    fmt.Println("مراكز المجموعات:")
    for i, center := range centers.Data() {
        fmt.Printf("المجموعة %d: (%.2f, %.2f)\n", i+1, center[0], center[1])
    }
    
    fmt.Printf("تم تجميع %d نقطة إلى %d مجموعة\n", len(data), 2)
}
```

## الأنماط الشائعة

### نمط 1: تحضير البيانات

```go
// تحميل البيانات من ملف CSV (مثال مبسط)
func loadDataFromCSV(filename string) (core.Tensor, core.Tensor, error) {
    // هذا مثال مبسط - في الواقع ستحتاج لمكتبة CSV
    features := [][]float64{
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0},
    }
    labels := [][]float64{
        {0}, {1}, {0},
    }
    
    return core.EasyTensor(features), core.EasyTensor(labels), nil
}

// تطبيع البيانات
func preprocessData(X core.Tensor) core.Tensor {
    return preprocessing.EasyStandardScale(X)
}

// تقسيم البيانات
func splitData(X, y core.Tensor) (core.Tensor, core.Tensor, core.Tensor, core.Tensor) {
    return preprocessing.EasySplit(X, y, 0.2)
}
```

### نمط 2: إنشاء نموذج شبكة عصبية

```go
func createNeuralNetwork(inputSize, hiddenSize, outputSize int) *models.Sequential {
    model := models.NewSequential()
    
    // الطبقة المخفية
    model.AddLayer(layers.NewDense(hiddenSize, activations.NewReLU()))
    
    // طبقة الإخراج
    if outputSize == 1 {
        // للتصنيف الثنائي
        model.AddLayer(layers.NewDense(outputSize, activations.NewSigmoid()))
    } else {
        // للتصنيف متعدد الفئات
        model.AddLayer(layers.NewDense(outputSize, activations.NewSoftmax()))
    }
    
    return model
}
```

### نمط 3: تدريب وتقييم النموذج

```go
func trainAndEvaluate(model *models.Sequential, XTrain, yTrain, XTest, yTest core.Tensor) error {
    // تجميع النموذج
    model.Compile(optimizers.NewAdam(0.001), losses.NewBinaryCrossEntropy())
    
    // التدريب
    history, err := model.EasyTrain(XTrain, yTrain)
    if err != nil {
        return fmt.Errorf("خطأ في التدريب: %v", err)
    }
    
    // التنبؤ
    predictions, err := model.EasyPredict(XTest)
    if err != nil {
        return fmt.Errorf("خطأ في التنبؤ: %v", err)
    }
    
    // التقييم
    accuracy := metrics.Accuracy(yTest, predictions)
    
    fmt.Printf("آخر خسارة: %.4f\n", history.Loss[len(history.Loss)-1])
    fmt.Printf("دقة النموذج: %.2f%%\n", accuracy*100)
    
    return nil
}
```

## أفضل الممارسات

### 1. إدارة البيانات

```go
// ✅ جيد: التحقق من صحة البيانات
func validateData(X, y core.Tensor) error {
    if X.Shape()[0] != y.Shape()[0] {
        return fmt.Errorf("عدد العينات في X (%d) لا يطابق عدد التسميات في y (%d)", 
                         X.Shape()[0], y.Shape()[0])
    }
    return nil
}

// ✅ جيد: تطبيع البيانات دائماً
func preprocessFeatures(X core.Tensor) core.Tensor {
    return preprocessing.EasyStandardScale(X)
}

// ❌ سيء: عدم التحقق من البيانات
func badExample(X, y core.Tensor) {
    // تدريب مباشر بدون تحقق أو تطبيع
    model := algorithms.EasyLinearRegression()
    model.Fit(X, y)  // قد يفشل!
}
```

### 2. إعداد النماذج

```go
// ✅ جيد: استخدام الدوال المبسطة للبداية
func goodModelSetup() *algorithms.LinearRegression {
    return algorithms.EasyLinearRegression()
}

// ✅ جيد: تخصيص المعاملات عند الحاجة
func customModelSetup() *algorithms.LinearRegression {
    return algorithms.NewLinearRegression(
        algorithms.WithLinearLearningRate(0.001),
        algorithms.WithLinearMaxIterations(2000),
    )
}
```

### 3. معالجة الأخطاء

```go
// ✅ جيد: التحقق من الأخطاء دائماً
func goodErrorHandling(model *models.Sequential, X, y core.Tensor) {
    history, err := model.EasyTrain(X, y)
    if err != nil {
        fmt.Printf("فشل التدريب: %v\n", err)
        return
    }
    
    predictions, err := model.EasyPredict(X)
    if err != nil {
        fmt.Printf("فشل التنبؤ: %v\n", err)
        return
    }
    
    // استخدام النتائج...
}

// ❌ سيء: تجاهل الأخطاء
func badErrorHandling(model *models.Sequential, X, y core.Tensor) {
    history, _ := model.EasyTrain(X, y)  // تجاهل الخطأ!
    predictions, _ := model.EasyPredict(X)  // خطر!
    // قد يؤدي إلى crash أو نتائج خاطئة
}
```

### 4. تنظيم الكود

```go
// ✅ جيد: تقسيم الكود إلى دوال منطقية
type MLPipeline struct {
    model *models.Sequential
}

func (p *MLPipeline) LoadData() (core.Tensor, core.Tensor, error) {
    // تحميل البيانات
    return nil, nil, nil
}

func (p *MLPipeline) PreprocessData(X core.Tensor) core.Tensor {
    // تحضير البيانات
    return preprocessing.EasyStandardScale(X)
}

func (p *MLPipeline) Train(X, y core.Tensor) error {
    // التدريب
    _, err := p.model.EasyTrain(X, y)
    return err
}

func (p *MLPipeline) Predict(X core.Tensor) (core.Tensor, error) {
    // التنبؤ
    return p.model.EasyPredict(X)
}
```

## استكشاف الأخطاء وإصلاحها

### الأخطاء الشائعة وحلولها

#### 1. خطأ: "النموذج غير مُجمع"

**الرسالة:**
```
خطأ: النموذج غير مُجمع - يجب استخدام Compile() قبل التدريب
```

**السبب:** محاولة تدريب نموذج شبكة عصبية بدون تجميعه أولاً.

**الحل:**
```go
// ❌ خطأ
model := models.NewSequential()
model.AddLayer(layers.NewDense(10, activations.NewReLU()))
history, err := model.EasyTrain(X, y)  // خطأ!

// ✅ صحيح
model := models.NewSequential()
model.AddLayer(layers.NewDense(10, activations.NewReLU()))
model.Compile(optimizers.NewAdam(0.01), losses.NewMeanSquaredError())  // تجميع أولاً
history, err := model.EasyTrain(X, y)  // يعمل بشكل صحيح
```

#### 2. خطأ: "أبعاد البيانات غير متطابقة"

**الرسالة:**
```
خطأ: شكل البيانات غير صحيح - متوقع [100, 2] لكن تم الحصول على [100, 3]
```

**السبب:** عدم تطابق أبعاد البيانات مع ما يتوقعه النموذج.

**الحل:**
```go
// التحقق من أبعاد البيانات
fmt.Printf("شكل X: %v\n", X.Shape())
fmt.Printf("شكل y: %v\n", y.Shape())

// التأكد من أن عدد الصفوف متطابق
if X.Shape()[0] != y.Shape()[0] {
    fmt.Printf("خطأ: عدد العينات غير متطابق\n")
    return
}

// التأكد من أن عدد الأعمدة صحيح للطبقة الأولى
inputSize := X.Shape()[1]
model.AddLayer(layers.NewDense(10, activations.NewReLU()))  // يجب أن تتطابق مع inputSize
```

#### 3. خطأ: "معدل التعلم مرتفع جداً"

**الأعراض:** الخسارة تزيد بدلاً من أن تقل، أو تصبح NaN.

**الحل:**
```go
// ❌ معدل تعلم مرتفع
model.Compile(optimizers.NewAdam(1.0), losses.NewMeanSquaredError())

// ✅ معدل تعلم مناسب
model.Compile(optimizers.NewAdam(0.001), losses.NewMeanSquaredError())

// أو استخدام الإعدادات الافتراضية
model.Compile(optimizers.NewAdam(0.01), losses.NewMeanSquaredError())
```

#### 4. خطأ: "البيانات غير مُطبعة"

**الأعراض:** التدريب بطيء جداً أو لا يتقارب.

**الحل:**
```go
// ✅ تطبيع البيانات دائماً
XScaled := preprocessing.EasyStandardScale(X)

// أو استخدام Min-Max scaling
XScaled := preprocessing.EasyMinMaxScale(X)

// ثم استخدام البيانات المُطبعة
history, err := model.EasyTrain(XScaled, y)
```

#### 5. خطأ: "نفاد الذاكرة"

**السبب:** البيانات كبيرة جداً أو batch size مرتفع.

**الحل:**
```go
// استخدام batch size أصغر
config := core.TrainingConfig{
    Epochs:    100,
    BatchSize: 16,  // بدلاً من 128
    Shuffle:   true,
}
history, err := model.Fit(X, y, config)

// أو استخدام الإعدادات الافتراضية المحسنة
history, err := model.EasyTrain(X, y)  // batch size = 32 افتراضياً
```

### نصائح لتحسين الأداء

#### 1. اختيار معدل التعلم المناسب

```go
// ابدأ بمعدل تعلم متوسط
optimizer := optimizers.NewAdam(0.01)

// إذا كان التدريب بطيئاً، زد المعدل
optimizer := optimizers.NewAdam(0.1)

// إذا كانت الخسارة تتذبذب، قلل المعدل
optimizer := optimizers.NewAdam(0.001)
```

#### 2. اختيار عدد العصور المناسب

```go
// راقب الخسارة لتحديد العدد المناسب
config := core.TrainingConfig{
    Epochs:  100,
    Verbose: 1,  // لعرض التقدم
}

// توقف عندما تستقر الخسارة
// إذا استمرت في التحسن، زد العدد
// إذا بدأت في الزيادة، قلل العدد أو استخدم early stopping
```

#### 3. تحسين هيكل الشبكة

```go
// للمشاكل البسيطة
model.AddLayer(layers.NewDense(10, activations.NewReLU()))
model.AddLayer(layers.NewDense(1, activations.NewSigmoid()))

// للمشاكل المعقدة
model.AddLayer(layers.NewDense(64, activations.NewReLU()))
model.AddLayer(layers.NewDense(32, activations.NewReLU()))
model.AddLayer(layers.NewDense(1, activations.NewSigmoid()))

// إضافة Dropout لمنع Overfitting
model.AddLayer(layers.NewDense(64, activations.NewReLU()))
model.AddLayer(layers.NewDropout(0.2))
model.AddLayer(layers.NewDense(32, activations.NewReLU()))
model.AddLayer(layers.NewDropout(0.2))
model.AddLayer(layers.NewDense(1, activations.NewSigmoid()))
```

### أدوات التشخيص

#### 1. مراقبة التدريب

```go
// تفعيل الوضع المفصل لمراقبة التقدم
config := core.TrainingConfig{
    Epochs:  100,
    Verbose: 1,  // عرض التقدم كل epoch
}

history, err := model.Fit(X, y, config)
if err != nil {
    return err
}

// تحليل تاريخ التدريب
fmt.Printf("الخسارة الأولى: %.4f\n", history.Loss[0])
fmt.Printf("الخسارة الأخيرة: %.4f\n", history.Loss[len(history.Loss)-1])

// التحقق من التحسن
improvement := history.Loss[0] - history.Loss[len(history.Loss)-1]
fmt.Printf("التحسن الإجمالي: %.4f\n", improvement)
```

#### 2. تقييم الأداء

```go
// للتصنيف
predictions, _ := model.EasyPredict(XTest)
accuracy := metrics.Accuracy(yTest, predictions)
precision := metrics.Precision(yTest, predictions)
recall := metrics.Recall(yTest, predictions)

fmt.Printf("الدقة: %.2f%%\n", accuracy*100)
fmt.Printf("الدقة المحددة: %.2f%%\n", precision*100)
fmt.Printf("الاستدعاء: %.2f%%\n", recall*100)

// للانحدار
mse := metrics.MeanSquaredError(yTest, predictions)
mae := metrics.MeanAbsoluteError(yTest, predictions)

fmt.Printf("متوسط مربع الخطأ: %.4f\n", mse)
fmt.Printf("متوسط الخطأ المطلق: %.4f\n", mae)
```

## مرجع الواجهات البرمجية

### الدوال المبسطة الجديدة

#### في models/sequential.go

```go
// تدريب مبسط مع إعدادات افتراضية جيدة
func (m *Sequential) EasyTrain(X, y core.Tensor) (*core.History, error)

// تنبؤ مبسط مع التحقق من الأخطاء
func (m *Sequential) EasyPredict(X core.Tensor) (core.Tensor, error)
```

#### في algorithms/

```go
// إنشاء نموذج انحدار خطي بإعدادات افتراضية
func EasyLinearRegression() *LinearRegression

// إنشاء نموذج انحدار لوجستي بإعدادات افتراضية
func EasyLogisticRegression() *LogisticRegression

// إنشاء نموذج K-Means بإعدادات افتراضية
func EasyKMeans(k int) *KMeans
```

#### في preprocessing/

```go
// تطبيع معياري سريع
func EasyStandardScale(X core.Tensor) core.Tensor

// تطبيع Min-Max سريع
func EasyMinMaxScale(X core.Tensor) core.Tensor

// تقسيم البيانات بسهولة
func EasySplit(X, y core.Tensor, testSize float64) (XTrain, XTest, yTrain, yTest core.Tensor)
```

#### في core/tensor.go

```go
// إنشاء tensor من slice بسهولة
func EasyTensor(data [][]float64) Tensor
```

### الإعدادات الافتراضية

#### للتدريب (EasyTrain)
- العصور: 50
- حجم الدفعة: 32
- نسبة التحقق: 0.2
- خلط البيانات: true
- الوضع المفصل: 1

#### للخوارزميات
- معدل التعلم: 0.01
- أقصى تكرار: 1000
- التسامح: 1e-6
- البذرة العشوائية: 42

### أمثلة سريعة

#### مثال كامل للتصنيف

```go
// تحميل البيانات
X, y := loadMyData()

// تقسيم البيانات
XTrain, XTest, yTrain, yTest := preprocessing.EasySplit(X, y, 0.2)

// تطبيع البيانات
XTrainScaled := preprocessing.EasyStandardScale(XTrain)
XTestScaled := preprocessing.EasyStandardScale(XTest)

// إنشاء النموذج
model := models.NewSequential()
model.AddLayer(layers.NewDense(10, activations.NewReLU()))
model.AddLayer(layers.NewDense(1, activations.NewSigmoid()))
model.Compile(optimizers.NewAdam(0.01), losses.NewBinaryCrossEntropy())

// التدريب
history, err := model.EasyTrain(XTrainScaled, yTrain)
if err != nil {
    log.Fatal(err)
}

// التنبؤ
predictions, err := model.EasyPredict(XTestScaled)
if err != nil {
    log.Fatal(err)
}

// التقييم
accuracy := metrics.Accuracy(yTest, predictions)
fmt.Printf("دقة النموذج: %.2f%%\n", accuracy*100)
```

#### مثال كامل للانحدار

```go
// البيانات
X, y := loadRegressionData()

// التحضير
XTrain, XTest, yTrain, yTest := preprocessing.EasySplit(X, y, 0.2)
XTrainScaled := preprocessing.EasyStandardScale(XTrain)
XTestScaled := preprocessing.EasyStandardScale(XTest)

// النموذج
model := algorithms.EasyLinearRegression()
err := model.Fit(XTrainScaled, yTrain)
if err != nil {
    log.Fatal(err)
}

// التنبؤ والتقييم
predictions, _ := model.Predict(XTestScaled)
mse := metrics.MeanSquaredError(yTest, predictions)
fmt.Printf("متوسط مربع الخطأ: %.4f\n", mse)
```

#### مثال كامل للتجميع

```go
// البيانات
X := loadClusteringData()

// النموذج
model := algorithms.EasyKMeans(3)  // 3 مجموعات
err := model.Fit(X)
if err != nil {
    log.Fatal(err)
}

// النتائج
labels := model.GetLabels()
centers := model.GetCenters()

fmt.Printf("تم تجميع البيانات إلى %d مجموعة\n", len(centers.Data()))
```

---

## خاتمة

هذا الدليل يغطي الأساسيات والمفاهيم المتقدمة لاستخدام مكتبة ThinkingNet-Go. المكتبة مصممة لتكون قوية ومرنة، مع توفير دوال مبسطة للمبتدئين وخيارات متقدمة للخبراء.

### الخطوات التالية

1. جرب الأمثلة الموجودة في مجلد `examples/`
2. اقرأ الكود المصدري لفهم التفاصيل الداخلية
3. ساهم في تطوير المكتبة على GitHub
4. انضم إلى مجتمع المطورين لطرح الأسئلة والمساعدة

### موارد إضافية

- [مستودع GitHub](https://github.com/blackmoon87/thinkingnet)
- [أمثلة إضافية](./examples/)
- [اختبارات الوحدة](./pkg/)
- [دليل المساهمة](./CONTRIBUTING.md)

**نتمنى لك تجربة ممتعة ومثمرة مع ThinkingNet-Go!** 🚀