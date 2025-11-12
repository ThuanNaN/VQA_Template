# Template Improvement Recommendations for Vietnamese VQA Datasets

This document provides concrete template improvements and examples for Vietnamese VQA datasets, based on the comprehensive survey analysis.

## Table of Contents
1. [ViVQA Template Enhancements](#vivqa-template-enhancements)
2. [EVJVQA Vietnamese E-commerce Templates](#evjvqa-vietnamese-e-commerce-templates)
3. [ViCLEVR Natural Vietnamese Templates](#viclevr-natural-vietnamese-templates)
4. [ViOCRVQA/ViTextVQA Template Extensions](#viocrvqavitextvqa-template-extensions)
5. [Implementation Guidelines](#implementation-guidelines)

---

## ViVQA Template Enhancements

### Current State
ViVQA currently uses basic templates focusing on object recognition. These need expansion to cover more complex reasoning.

### Proposed Template Categories

#### 1. Spatial Reasoning Templates (10 types)

**Template Type: Positional Queries**
```
English: "What is to the {left/right} of the {object}?"
Vietnamese: "Cái gì ở bên {trái/phải} của {object}?"

Examples:
- "Cái gì ở bên trái của chiếc xe?"
- "Có gì ở phía sau người đàn ông?"
- "Vật gì nằm giữa cái bàn và cái ghế?"
```

**Template Type: Relative Position**
```
English: "Where is the {object} located?"
Vietnamese: "{Object} nằm ở đâu?"

Examples:
- "Con chó nằm ở đâu trong bức tranh?"
- "Người phụ nữ đứng ở vị trí nào?"
- "Chiếc ô tô được đỗ ở chỗ nào?"
```

**Template Type: Direction and Orientation**
```
English: "Which direction is the {object} facing?"
Vietnamese: "{Object} đang hướng về phía nào?"

Examples:
- "Con ngựa đang nhìn về hướng nào?"
- "Người đàn ông đang quay mặt về đâu?"
```

**Template Type: Distance/Proximity**
```
English: "Is the {object1} near the {object2}?"
Vietnamese: "{Object1} có gần {object2} không?"

Examples:
- "Chiếc xe có gần cái cây không?"
- "Những người này có đứng gần nhau không?"
```

**Template Type: Inside/Outside**
```
English: "What is inside/outside the {container}?"
Vietnamese: "Có gì {trong/ngoài} {container}?"

Examples:
- "Có gì trong cái hộp?"
- "Những gì ở ngoài tòa nhà?"
```

**Template Type: Above/Below**
```
English: "What is {above/below} the {object}?"
Vietnamese: "Cái gì ở {trên/dưới} {object}?"

Examples:
- "Có gì ở trên bàn?"
- "Cái gì nằm dưới cái ghế?"
```

**Template Type: Between Objects**
```
English: "What is between the {object1} and {object2}?"
Vietnamese: "Có gì ở giữa {object1} và {object2}?"

Examples:
- "Có gì ở giữa hai chiếc xe?"
- "Vật gì nằm giữa người đàn ông và người phụ nữ?"
```

**Template Type: Around/Surrounding**
```
English: "What is around the {object}?"
Vietnamese: "Có gì xung quanh {object}?"

Examples:
- "Có gì xung quanh ngôi nhà?"
- "Những gì ở quanh cái bàn?"
```

**Template Type: In Front/Behind**
```
English: "What is in front of/behind the {object}?"
Vietnamese: "Có gì ở {phía trước/phía sau} {object}?"

Examples:
- "Có gì ở phía trước cửa hàng?"
- "Cái gì đằng sau người đàn ông?"
```

**Template Type: Corner/Edge**
```
English: "What is in the {corner/edge} of the image?"
Vietnamese: "Có gì ở {góc/cạnh} của ảnh?"

Examples:
- "Có gì ở góc trên bên phải?"
- "Vật gì ở cạnh dưới của bức ảnh?"
```

#### 2. Counting Templates with Conditions (5 types)

**Template Type: Color-Based Counting**
```
English: "How many {color} {objects} are there?"
Vietnamese: "Có bao nhiêu {object} màu {color}?"

Examples:
- "Có bao nhiêu chiếc xe màu đỏ?"
- "Có mấy người mặc áo xanh?"
```

**Template Type: Action-Based Counting**
```
English: "How many people are {action}?"
Vietnamese: "Có bao nhiêu người đang {action}?"

Examples:
- "Có bao nhiêu người đang ngồi?"
- "Có mấy con chó đang chạy?"
```

**Template Type: Attribute-Based Counting**
```
English: "How many {adjective} {objects} are visible?"
Vietnamese: "Có bao nhiêu {object} {adjective}?"

Examples:
- "Có bao nhiêu cái cốc lớn?"
- "Có mấy chiếc xe nhỏ?"
```

**Template Type: Comparative Counting**
```
English: "Are there more {object1} than {object2}?"
Vietnamese: "{Object1} có nhiều hơn {object2} không?"

Examples:
- "Người đàn ông có nhiều hơn người phụ nữ không?"
- "Xe ô tô có nhiều hơn xe máy không?"
```

**Template Type: Exact/Range Counting**
```
English: "Is/Are there {number} {object}?"
Vietnamese: "Có phải có {number} {object} không?"

Examples:
- "Có phải có 3 con mèo không?"
- "Có đúng 5 người trong ảnh không?"
```

#### 3. Comparative Questions (5 types)

**Template Type: Size Comparison**
```
English: "Which is {larger/smaller}, the {object1} or the {object2}?"
Vietnamese: "{Object1} hay {object2} {lớn hơn/nhỏ hơn}?"

Examples:
- "Con chó hay con mèo lớn hơn?"
- "Cái bàn hay cái ghế nhỏ hơn?"
```

**Template Type: Color Comparison**
```
English: "Is the {object1} the same color as the {object2}?"
Vietnamese: "{Object1} có cùng màu với {object2} không?"

Examples:
- "Chiếc xe có cùng màu với tòa nhà không?"
- "Cái áo có màu giống cái quần không?"
```

**Template Type: Quantity Comparison**
```
English: "Which has more, {group1} or {group2}?"
Vietnamese: "{Group1} hay {group2} nhiều hơn?"

Examples:
- "Người đàn ông hay người phụ nữ nhiều hơn?"
- "Cái ghế hay cái bàn nhiều hơn?"
```

**Template Type: Attribute Comparison**
```
English: "Which {object} is {attribute}?"
Vietnamese: "{Object} nào {attribute} hơn?"

Examples:
- "Chiếc xe nào cao hơn?"
- "Người nào trẻ hơn?"
```

**Template Type: Multiple Object Comparison**
```
English: "Which is the {superlative} {object}?"
Vietnamese: "{Object} nào {superlative} nhất?"

Examples:
- "Tòa nhà nào cao nhất?"
- "Con vật nào nhỏ nhất?"
```

#### 4. Attribute Combination Templates (10 types)

**Template Type: Color + Size**
```
English: "What color is the {largest/smallest} {object}?"
Vietnamese: "{Object} {lớn nhất/nhỏ nhất} có màu gì?"

Examples:
- "Con chó lớn nhất có màu gì?"
- "Chiếc xe nhỏ nhất là màu gì?"
```

**Template Type: Color + Position**
```
English: "What is the color of the {object} on the {position}?"
Vietnamese: "{Object} ở {position} có màu gì?"

Examples:
- "Chiếc xe ở bên trái có màu gì?"
- "Cái bàn ở giữa là màu gì?"
```

**Template Type: Size + Position**
```
English: "Where is the {largest/smallest} {object}?"
Vietnamese: "{Object} {lớn nhất/nhỏ nhất} ở đâu?"

Examples:
- "Con chó lớn nhất ở đâu?"
- "Cái cây nhỏ nhất nằm ở vị trí nào?"
```

**Template Type: Action + Color**
```
English: "What color is the {object} that is {action}?"
Vietnamese: "{Object} đang {action} có màu gì?"

Examples:
- "Con chó đang chạy có màu gì?"
- "Người đang ngồi mặc áo màu gì?"
```

**Template Type: Count + Color**
```
English: "How many {color} {objects} are {action/position}?"
Vietnamese: "Có bao nhiêu {object} màu {color} đang {action/position}?"

Examples:
- "Có bao nhiêu chiếc xe màu đỏ đang đỗ?"
- "Có mấy con chó màu nâu đang ngồi?"
```

**Template Type: Material + Color**
```
English: "What color is the {material} {object}?"
Vietnamese: "{Object} bằng {material} có màu gì?"

Examples:
- "Cái bàn bằng gỗ có màu gì?"
- "Tòa nhà bằng kính là màu gì?"
```

**Template Type: Shape + Color**
```
English: "What color is the {shape} {object}?"
Vietnamese: "{Object} hình {shape} có màu gì?"

Examples:
- "Vật hình tròn có màu gì?"
- "Cái hộp hình vuông là màu gì?"
```

**Template Type: Multiple Attributes**
```
English: "Is there a {color} {size} {object} {position}?"
Vietnamese: "Có {object} {size} màu {color} ở {position} không?"

Examples:
- "Có chiếc xe lớn màu đỏ ở bên trái không?"
- "Có con chó nhỏ màu nâu đang ngồi không?"
```

**Template Type: Attribute Chain**
```
English: "What {attribute2} is the {attribute1} {object}?"
Vietnamese: "{Object} {attribute1} có {attribute2} như thế nào?"

Examples:
- "Con chó lớn có màu gì?"
- "Chiếc xe đỏ có kích thước ra sao?"
```

**Template Type: Conditional Attributes**
```
English: "Among the {objects}, which one is {attribute}?"
Vietnamese: "Trong số các {object}, cái nào {attribute}?"

Examples:
- "Trong số các chiếc xe, cái nào màu đỏ?"
- "Trong các con vật, con nào lớn nhất?"
```

#### 5. Causal Reasoning Templates (5 types)

**Template Type: Why Questions**
```
English: "Why is the {object} {state/action}?"
Vietnamese: "Tại sao {object} {state/action}?"

Examples:
- "Tại sao người đàn ông đang cười?"
- "Vì sao con chó đang chạy?"
```

**Template Type: Purpose Questions**
```
English: "What is the {object} used for?"
Vietnamese: "{Object} được dùng để làm gì?"

Examples:
- "Cái này được dùng để làm gì?"
- "Vật dụng đó có mục đích gì?"
```

**Template Type: Effect Questions**
```
English: "What will happen if {condition}?"
Vietnamese: "Điều gì sẽ xảy ra nếu {condition}?"

Examples:
- "Điều gì sẽ xảy ra nếu trời mưa?"
- "Chuyện gì sẽ xảy ra nếu cửa đóng lại?"
```

**Template Type: Cause Questions**
```
English: "What caused the {effect}?"
Vietnamese: "Điều gì gây ra {effect}?"

Examples:
- "Điều gì làm cho con chó vui vẻ?"
- "Nguyên nhân nào khiến người đàn ông ngã?"
```

**Template Type: Consequence Questions**
```
English: "What is the result of {action}?"
Vietnamese: "Kết quả của {action} là gì?"

Examples:
- "Kết quả của việc đóng cửa là gì?"
- "Hậu quả của hành động đó là gì?"
```

---

## EVJVQA Vietnamese E-commerce Templates

### Current State
EVJVQA uses English questions. Need comprehensive Vietnamese templates specific to e-commerce context.

### Vietnamese Shopping Context Templates

#### 1. Product Attribute Templates (20 types)

**Template Type: Color Questions**
```
Vietnamese: "{Sản phẩm} này có màu gì?"
Vietnamese: "Màu sắc của {sản phẩm} là gì?"
Vietnamese: "{Sản phẩm} có những màu nào?"

Examples:
- "Chiếc áo này có màu gì?"
- "Đôi giày có những màu nào?"
- "Màu sắc của chiếc váy là gì?"
```

**Template Type: Size Questions**
```
Vietnamese: "{Sản phẩm} có size nào?"
Vietnamese: "Kích cỡ của {sản phẩm} là bao nhiêu?"
Vietnamese: "{Sản phẩm} này có những size gì?"

Examples:
- "Áo này có size nào?"
- "Giày có size 39 không?"
- "Quần có những size nào?"
```

**Template Type: Material Questions**
```
Vietnamese: "{Sản phẩm} được làm từ chất liệu gì?"
Vietnamese: "Chất liệu của {sản phẩm} là gì?"
Vietnamese: "{Sản phẩm} làm bằng gì?"

Examples:
- "Chiếc áo làm bằng vải gì?"
- "Đôi giày được làm từ chất liệu gì?"
```

**Template Type: Price Questions**
```
Vietnamese: "{Sản phẩm} giá bao nhiêu?"
Vietnamese: "Giá của {sản phẩm} là bao nhiêu?"
Vietnamese: "{Sản phẩm} này bán với giá nào?"

Examples:
- "Chiếc áo này giá bao nhiêu?"
- "Đôi giày có giá bao nhiêu?"
```

**Template Type: Brand Questions**
```
Vietnamese: "{Sản phẩm} là nhãn hiệu gì?"
Vietnamese: "Thương hiệu của {sản phẩm} là gì?"
Vietnamese: "{Sản phẩm} thuộc hãng nào?"

Examples:
- "Chiếc áo là nhãn hiệu gì?"
- "Đôi giày thuộc thương hiệu nào?"
```

**Template Type: Style Questions**
```
Vietnamese: "{Sản phẩm} thuộc kiểu dáng nào?"
Vietnamese: "Phong cách của {sản phẩm} là gì?"
Vietnamese: "{Sản phẩm} có thiết kế như thế nào?"

Examples:
- "Chiếc váy thuộc kiểu dáng nào?"
- "Áo có phong cách gì?"
```

**Template Type: Availability**
```
Vietnamese: "{Sản phẩm} còn hàng không?"
Vietnamese: "Có còn {sản phẩm} này không?"
Vietnamese: "{Sản phẩm} {màu/size} còn không?"

Examples:
- "Áo màu đỏ size M còn không?"
- "Có còn đôi giày này không?"
```

**Template Type: Pattern/Design**
```
Vietnamese: "{Sản phẩm} có họa tiết gì?"
Vietnamese: "Họa tiết của {sản phẩm} là gì?"
Vietnamese: "{Sản phẩm} có hoa văn như thế nào?"

Examples:
- "Chiếc áo có họa tiết gì?"
- "Váy có hoa văn không?"
```

**Template Type: Features**
```
Vietnamese: "{Sản phẩm} có tính năng gì?"
Vietnamese: "Đặc điểm của {sản phẩm} là gì?"
Vietnamese: "{Sản phẩm} có những ưu điểm nào?"

Examples:
- "Chiếc áo có đặc điểm gì nổi bật?"
- "Đôi giày có tính năng gì đặc biệt?"
```

**Template Type: Warranty/Return**
```
Vietnamese: "{Sản phẩm} có bảo hành không?"
Vietnamese: "{Sản phẩm} có thể đổi trả không?"
Vietnamese: "Chính sách bảo hành của {sản phẩm} như thế nào?"

Examples:
- "Sản phẩm này có bảo hành không?"
- "Có thể đổi trả áo này không?"
```

#### 2. Comparison Templates (10 types)

**Template Type: Price Comparison**
```
Vietnamese: "{Sản phẩm 1} hay {sản phẩm 2} rẻ hơn?"
Vietnamese: "Cái nào có giá tốt hơn?"
Vietnamese: "{Sản phẩm 1} và {sản phẩm 2}, cái nào đắt hơn?"

Examples:
- "Chiếc áo đỏ hay áo xanh rẻ hơn?"
- "Đôi giày nào có giá tốt hơn?"
```

**Template Type: Quality Comparison**
```
Vietnamese: "{Sản phẩm 1} hay {sản phẩm 2} chất lượng hơn?"
Vietnamese: "Sản phẩm nào tốt hơn?"
Vietnamese: "Cái nào bền hơn?"

Examples:
- "Áo cotton hay áo polyester bền hơn?"
- "Giày da hay giày vải chất lượng hơn?"
```

**Template Type: Size Comparison**
```
Vietnamese: "{Sản phẩm 1} hay {sản phẩm 2} to hơn?"
Vietnamese: "Cái nào có kích cỡ lớn hơn?"

Examples:
- "Chiếc áo nào to hơn?"
- "Túi nào có size lớn hơn?"
```

**Template Type: Color Options Comparison**
```
Vietnamese: "{Sản phẩm 1} hay {sản phẩm 2} có nhiều màu hơn?"
Vietnamese: "Sản phẩm nào có đa dạng màu sắc hơn?"

Examples:
- "Áo này hay áo kia có nhiều màu hơn?"
- "Dòng sản phẩm nào đa dạng màu sắc hơn?"
```

**Template Type: Value Comparison**
```
Vietnamese: "{Sản phẩm 1} hay {sản phẩm 2} đáng mua hơn?"
Vietnamese: "Nên mua sản phẩm nào?"

Examples:
- "Nên mua áo nào?"
- "Giày nào đáng tiền hơn?"
```

#### 3. Recommendation Templates (5 types)

**Template Type: Suitability**
```
Vietnamese: "{Sản phẩm} có phù hợp với {mục đích} không?"
Vietnamese: "{Sản phẩm} có thích hợp cho {người/trường hợp} không?"

Examples:
- "Áo này có phù hợp đi làm không?"
- "Giày này có thích hợp cho trẻ em không?"
```

**Template Type: Occasion**
```
Vietnamese: "{Sản phẩm} phù hợp với dịp nào?"
Vietnamese: "Nên mặc {sản phẩm} này vào lúc nào?"

Examples:
- "Chiếc váy này phù hợp với dịp nào?"
- "Nên mặc áo này vào dịp gì?"
```

**Template Type: Matching**
```
Vietnamese: "{Sản phẩm 1} có hợp với {sản phẩm 2} không?"
Vietnamese: "Nên kết hợp {sản phẩm 1} với gì?"

Examples:
- "Chiếc áo này có hợp với quần jeans không?"
- "Nên kết hợp giày này với loại quần gì?"
```

**Template Type: Season Suitability**
```
Vietnamese: "{Sản phẩm} có phù hợp với mùa {season} không?"
Vietnamese: "{Sản phẩm} nên mặc vào mùa nào?"

Examples:
- "Áo này có phù hợp với mùa hè không?"
- "Nên mặc áo khoác này vào mùa nào?"
```

**Template Type: Body Type**
```
Vietnamese: "{Sản phẩm} có phù hợp với {body type} không?"
Vietnamese: "{Sản phẩm} này có vừa với người {description} không?"

Examples:
- "Váy này có phù hợp với người mập không?"
- "Áo này có vừa với người cao không?"
```

#### 4. Vietnamese Measurement Terms

**Size Standards**:
- S (Nhỏ), M (Vừa), L (Lớn), XL (Rất lớn), XXL (Siêu lớn)
- Số: 35, 36, 37, 38, 39, 40 (giày)
- Chiều dài, chiều rộng, chiều cao

**Shopping Terms**:
- Giảm giá, khuyến mãi, sale off
- Miễn phí vận chuyển, freeship
- Hàng chính hãng, hàng cao cấp
- Đặt hàng, mua ngay, thêm vào giỏ
- Có sẵn, hết hàng, sắp về

---

## ViCLEVR Natural Vietnamese Templates

### Goal
Ensure CLEVR-style questions sound natural in Vietnamese while maintaining systematic coverage.

### Vietnamese Spatial Terms

**Common Spatial Relations**:
- Trái/phải (left/right)
- Trước/sau (front/back)
- Trên/dưới (above/below)
- Trong/ngoài (inside/outside)
- Gần/xa (near/far)
- Giữa (between)
- Xung quanh (around)

### Natural Vietnamese CLEVR Templates

#### Counting Questions
```
English: "How many objects are there?"
Vietnamese Natural: "Có bao nhiêu vật thể?"
Vietnamese Alternative: "Có mấy đồ vật trong ảnh?"

English: "How many red cubes are there?"
Vietnamese Natural: "Có bao nhiêu khối vuông màu đỏ?"
Vietnamese Alternative: "Có mấy hình khối màu đỏ?"
```

#### Existence Questions
```
English: "Is there a red cube?"
Vietnamese Natural: "Có khối vuông màu đỏ không?"
Vietnamese Alternative: "Có hình khối màu đỏ nào không?"

English: "Are there any metallic spheres?"
Vietnamese Natural: "Có quả cầu kim loại nào không?"
Vietnamese Alternative: "Có hình cầu bằng kim loại không?"
```

#### Comparison Questions
```
English: "Is the red cube larger than the blue sphere?"
Vietnamese Natural: "Khối vuông đỏ có lớn hơn quả cầu xanh không?"
Vietnamese Alternative: "Hình khối đỏ to hơn hình cầu xanh phải không?"

English: "Which is larger, the cube or the sphere?"
Vietnamese Natural: "Cái nào lớn hơn, khối vuông hay quả cầu?"
Vietnamese Alternative: "Hình vuông hay hình tròn to hơn?"
```

#### Attribute Questions
```
English: "What color is the large cube?"
Vietnamese Natural: "Khối vuông lớn có màu gì?"
Vietnamese Alternative: "Hình khối to nhất là màu gì?"

English: "What material is the small sphere?"
Vietnamese Natural: "Quả cầu nhỏ làm bằng chất liệu gì?"
Vietnamese Alternative: "Hình cầu nhỏ được làm từ gì?"
```

---

## ViOCRVQA/ViTextVQA Template Extensions

### Text-Based Reasoning Templates

#### Simple Text Reading
```
Vietnamese: "Có gì viết trên {object}?"
Vietnamese: "Dòng chữ trên {object} là gì?"
Vietnamese: "{Object} có nội dung gì?"

Examples:
- "Có gì viết trên biển báo?"
- "Dòng chữ trên sách là gì?"
```

#### Text Comprehension
```
Vietnamese: "Thông tin trên {object} cho biết điều gì?"
Vietnamese: "Nội dung {object} nói về gì?"
Vietnamese: "{Text} có nghĩa là gì?"

Examples:
- "Thông tin trên nhãn sản phẩm cho biết điều gì?"
- "Biển báo này nói về gì?"
```

#### Multi-step Text Reasoning
```
Vietnamese: "Dựa vào {text 1} và {text 2}, {question}?"
Vietnamese: "Kết hợp thông tin từ {source 1} và {source 2}, {question}?"

Examples:
- "Dựa vào giá và khối lượng, sản phẩm nào rẻ hơn?"
- "Kết hợp thông tin từ hai biển báo, đường nào gần hơn?"
```

#### Text Relationship
```
Vietnamese: "{Text 1} và {text 2} có liên quan như thế nào?"
Vietnamese: "Mối quan hệ giữa {text 1} và {text 2} là gì?"

Examples:
- "Tiêu đề và nội dung có liên quan như thế nào?"
- "Tên tác giả và tên sách có quan hệ gì?"
```

---

## Implementation Guidelines

### Template Design Principles

1. **Natural Language**:
   - Use common Vietnamese expressions
   - Avoid direct translations from English
   - Consider Vietnamese word order and grammar

2. **Consistency**:
   - Maintain consistent terminology across templates
   - Use standard Vietnamese punctuation (?, .)
   - Keep similar templates structurally aligned

3. **Diversity**:
   - Provide multiple variations for each template type
   - Include formal and informal versions where appropriate
   - Cover different ways to ask the same question

4. **Cultural Relevance**:
   - Use Vietnamese-specific measurements and terms
   - Include context relevant to Vietnamese users
   - Consider Vietnamese shopping habits for e-commerce

### Implementation Steps

1. **Template Library Creation**:
   ```python
   templates = {
       "spatial": {
           "position": [
               "Cái gì ở bên {direction} của {object}?",
               "{Object} nằm ở đâu?",
               # More variations...
           ],
           # More spatial types...
       },
       "counting": {
           "basic": ["Có bao nhiêu {object}?"],
           "conditional": ["Có bao nhiêu {object} màu {color}?"],
           # More counting types...
       },
       # More categories...
   }
   ```

2. **Template Validation**:
   - Review by native Vietnamese speakers
   - Test naturalness scores
   - Verify grammatical correctness

3. **Integration**:
   - Add to dataset generation pipelines
   - Create template selection algorithms
   - Implement template filling mechanisms

4. **Quality Assurance**:
   - Generate sample questions
   - Manual review of outputs
   - Automated diversity metrics

### Template Usage Examples

**For ViVQA**:
```python
# Generate spatial reasoning question
template = "Cái gì ở bên {direction} của {object}?"
question = template.format(
    direction="trái",
    object="chiếc xe"
)
# Result: "Cái gì ở bên trái của chiếc xe?"
```

**For EVJVQA**:
```python
# Generate e-commerce question
template = "{Sản phẩm} có size nào?"
question = template.format(
    sản_phẩm="Chiếc áo này"
)
# Result: "Chiếc áo này có size nào?"
```

### Metrics for Template Quality

1. **Diversity Score**: Measure unique templates per category
2. **Naturalness Score**: Native speaker ratings (1-5)
3. **Coverage**: Percentage of reasoning types covered
4. **Usability**: Successful question generation rate

---

## Next Steps

1. **Immediate**:
   - Implement template library structure
   - Add top 20 most important templates
   - Validate with native speakers

2. **Short-term**:
   - Expand to all template categories
   - Integrate with existing dataset loaders
   - Create template usage examples

3. **Long-term**:
   - Develop automatic template selection
   - Build template quality metrics
   - Create template contribution guidelines

---

**Version**: 1.0  
**Last Updated**: 2025-11-12  
**Contributors**: Dataset Template Enhancement Team
