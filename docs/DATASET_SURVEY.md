# Vietnamese VQA Datasets Survey

This document provides a comprehensive survey of Vietnamese Visual Question Answering (VQA) datasets, analyzing their construction methodologies, generation approaches, quality assessments, and potential contributions.

## Table of Contents
1. [ViVQA](#vivqa)
2. [OpenViVQA](#openvivqa)
3. [ViTextVQA](#vitextvqa)
4. [ViOCRVQA](#viocrvqa)
5. [ViCLEVR](#viclevr)
6. [EVJVQA](#evjvqa)
7. [Comparative Analysis](#comparative-analysis)
8. [Recommendations and Contributions](#recommendations-and-contributions)

---

## ViVQA

### Overview
**ViVQA** (Vietnamese Visual Question Answering) is one of the first Vietnamese VQA datasets, providing a foundation for Vietnamese language VQA research.

### Dataset Building Methodology
- **Source Images**: Uses MS COCO images (Common Objects in Context)
- **Question Generation**: 
  - **Method**: Template-based approach combined with human annotation
  - Questions are created by translating and adapting English VQA questions to Vietnamese
  - Human annotators formulate natural Vietnamese questions based on images
- **Answer Collection**: Human-annotated answers in Vietnamese
- **Dataset Size**: Contains training and test splits with thousands of QA pairs

### Generation Quality Assessment
- **Strengths**:
  - Native Vietnamese speaker annotations ensure linguistic accuracy
  - Leverages well-established COCO image dataset
  - Questions cover diverse visual concepts
  
- **Limitations**:
  - Template-based approach may limit question diversity
  - Translation-based questions might not reflect natural Vietnamese question patterns
  - Limited domain coverage (general object recognition)

### Template Method Analysis
- Uses question templates that are filled with detected objects/attributes
- Templates ensure grammatical correctness but may reduce naturalness
- **Potential Improvements**:
  - Expand template diversity to cover more question types
  - Add more complex reasoning questions beyond object recognition
  - Include spatial relationship and counting templates

### Human Labeling Effort
- **High human involvement**: Requires native Vietnamese speakers for both question creation and answer annotation
- Quality control through multiple annotation rounds
- Significant time and resource investment

---

## OpenViVQA

### Overview
**OpenViVQA** is an open-domain Vietnamese VQA dataset designed for more diverse and complex visual reasoning tasks.

### Dataset Building Methodology
- **Source Images**: Web-crawled images from diverse Vietnamese sources
- **Question Generation**:
  - **Method**: Human-generated questions by Vietnamese annotators
  - Focuses on open-domain scenarios relevant to Vietnamese context
  - Questions designed to require deeper reasoning beyond simple object recognition
- **Answer Processing**: 
  - Implements answer normalization to remove redundant question sequences
  - Special processing method: `remove_continuous_sequences` (as seen in codebase)
- **Dataset Size**: Structured with train, dev, and test splits

### Generation Quality Assessment
- **Strengths**:
  - More natural questions reflecting Vietnamese language patterns
  - Open-domain approach increases diversity
  - Context-aware questions specific to Vietnamese culture and scenarios
  - Advanced answer processing reduces redundancy
  
- **Limitations**:
  - Fully manual annotation is time-consuming and expensive
  - Potential inconsistencies across different annotators
  - Limited scalability due to high annotation cost

### Template Method Analysis
- **Minimal template usage**: Primarily human-generated free-form questions
- More natural language patterns but harder to scale
- **Potential Improvements**:
  - Could benefit from semi-automatic question generation to scale up
  - Hybrid approach combining templates for common patterns and manual annotation for complex cases

### Human Labeling Effort
- **Very high human involvement**: Entirely human-annotated questions and answers
- Requires significant expertise and native language understanding
- Most resource-intensive among the datasets reviewed

---

## ViTextVQA

### Overview
**ViTextVQA** extends Vietnamese VQA to include text reading and understanding within images, addressing the challenge of scene text in Vietnamese contexts.

### Dataset Building Methodology
- **Source Images**: Images containing Vietnamese text (signs, documents, product labels, etc.)
- **Question Generation**:
  - **Method**: Human annotation focusing on text-based reasoning
  - Questions specifically designed to test OCR and reading comprehension
  - Requires understanding both visual content and textual information
- **Answer Collection**: Multiple valid answers per question (as indicated by dataset structure)
- **Special Focus**: Scene text recognition and comprehension in Vietnamese

### Generation Quality Assessment
- **Strengths**:
  - Addresses critical gap in Vietnamese text understanding
  - Multiple answers per question capture answer variations
  - Realistic scenarios with Vietnamese text in natural contexts
  - Essential for practical applications (document understanding, sign reading)
  
- **Limitations**:
  - Requires images with clear, readable Vietnamese text
  - Annotation requires both visual and textual understanding
  - More complex annotation process than standard VQA

### Template Method Analysis
- **Hybrid approach**:
  - Some questions follow patterns (e.g., "What does the sign say?", "What is written on...?")
  - Other questions require contextual understanding beyond templates
- **Potential Improvements**:
  - Develop templates for common text-based question types
  - Add more questions combining text and visual reasoning
  - Include questions about text relationships and context

### Human Labeling Effort
- **High complexity**: Annotators need to:
  - Read and understand Vietnamese text in images
  - Formulate relevant questions
  - Provide accurate answers considering text content
- Quality assurance is crucial for text accuracy
- Moderate to high resource requirement

---

## ViOCRVQA

### Overview
**ViOCRVQA** (Vietnamese OCR-based VQA) focuses specifically on visual question answering requiring OCR capabilities in Vietnamese contexts.

### Dataset Building Methodology
- **Source Images**: Documents, books, products, and other text-rich images
- **Question Generation**:
  - **Method**: Human-annotated questions requiring text reading
  - Emphasis on OCR-dependent questions
  - Questions test both text extraction and comprehension
- **Answer Collection**: Multiple possible answers to account for variations
- **Image Sources**: Diverse text-containing scenes (book covers, product packaging, documents)

### Generation Quality Assessment
- **Strengths**:
  - Specialized focus on OCR requirements
  - Highly relevant for document understanding applications
  - Multiple answers increase robustness
  - Covers diverse text scenarios
  
- **Limitations**:
  - Narrow focus on text-based questions
  - Requires high-quality text images
  - May not generalize to non-text VQA scenarios

### Template Method Analysis
- **Template-friendly domain**:
  - Many OCR questions follow predictable patterns
  - Common templates: "Who is the author?", "What is the title?", "What is written on...?"
- **Potential Improvements**:
  - Expand beyond simple text extraction to text reasoning
  - Add questions about text relationships (comparison, inference)
  - Include multi-step reasoning over text content

### Human Labeling Effort
- **Moderate to high**: 
  - Requires reading and understanding Vietnamese text
  - Simpler than open-domain VQA but requires accuracy
  - Text verification adds quality control overhead

---

## ViCLEVR

### Overview
**ViCLEVR** is expected to be the Vietnamese adaptation of the CLEVR (Compositional Language and Elementary Visual Reasoning) dataset, focusing on systematic compositional reasoning.

### Dataset Building Methodology
- **Source Images**: Synthetically generated 3D rendered scenes with simple objects
- **Question Generation**:
  - **Method**: Template-based with systematic combinatorial approach
  - Questions follow structured patterns testing specific reasoning skills
  - Covers: counting, comparison, spatial reasoning, attribute identification
- **Generation Process**: Likely automated using predefined templates and scene graphs
- **Controlled Environment**: Synthetic scenes allow precise control over complexity

### Generation Quality Assessment
- **Strengths**:
  - Systematic coverage of reasoning types
  - Eliminates visual ambiguity through synthetic scenes
  - Scalable through automated generation
  - Precise ground truth from scene graphs
  - Ideal for diagnostic evaluation of reasoning capabilities
  
- **Limitations**:
  - Synthetic images don't reflect real-world complexity
  - Limited visual diversity
  - May not transfer well to natural images
  - Template-based questions are predictable

### Template Method Analysis
- **Pure template-based approach**:
  - Questions generated from compositional templates
  - Systematic variation of question types
  - Templates ensure balanced coverage of reasoning skills
- **Potential Improvements**:
  - Expand templates to cover more complex Vietnamese linguistic patterns
  - Add templates for Vietnamese-specific spatial/relational terms
  - Ensure templates produce natural-sounding Vietnamese questions
  - Validate that templates capture Vietnamese language nuances

### Human Labeling Effort
- **Minimal human effort**:
  - Mostly automated generation from scene graphs
  - Human effort mainly in template design and validation
  - Quality control through automated verification
- Most scalable approach among all datasets
- **Note**: Template refinement requires native Vietnamese speaker input to ensure naturalness

---

## EVJVQA

### Overview
**EVJVQA** (English-Vietnamese-Japanese VQA) is a multilingual VQA dataset designed for e-commerce domain, though currently focusing on English-Vietnamese pairs.

### Dataset Building Methodology
- **Source Images**: E-commerce product images
- **Question Generation**:
  - **Method**: Domain-specific question generation for e-commerce
  - Questions about product attributes, features, and characteristics
  - Currently uses English questions (as seen in dataset structure)
- **Domain**: Specialized for e-commerce applications
- **Multilingual Aspect**: Designed for cross-lingual VQA research

### Generation Quality Assessment
- **Strengths**:
  - Domain-specific focus on e-commerce
  - Practical applications for online shopping
  - Structured product-related questions
  - Multilingual potential for cross-lingual research
  
- **Limitations**:
  - Currently appears to be primarily English
  - Limited to e-commerce domain
  - May not generalize to other domains
  - Requires domain-specific annotation expertise

### Template Method Analysis
- **Highly template-friendly domain**:
  - E-commerce questions follow predictable patterns
  - Common templates: "What color is...", "What is the price of...", "What size..."
  - Product attributes lend themselves to structured questions
- **Potential Improvements**:
  - Develop Vietnamese templates for common e-commerce queries
  - Add comparison questions across products
  - Include complex reasoning (e.g., "Is this suitable for...")
  - Expand templates to cover Vietnamese shopping contexts

### Human Labeling Effort
- **Moderate effort**:
  - Domain knowledge required for relevant questions
  - Structured domain makes annotation more systematic
  - Can leverage product metadata to reduce annotation burden

---

## Comparative Analysis

### Dataset Building Approaches

| Dataset | Image Source | Question Method | Scalability | Complexity |
|---------|-------------|-----------------|-------------|------------|
| **ViVQA** | MS COCO | Template + Human | Medium | Medium |
| **OpenViVQA** | Web-crawled | Fully Human | Low | High |
| **ViTextVQA** | Text images | Human | Low | High |
| **ViOCRVQA** | Documents/Text | Human | Medium | Medium-High |
| **ViCLEVR** | Synthetic 3D | Automated Templates | Very High | Low-Medium |
| **EVJVQA** | E-commerce | Domain Templates | Medium-High | Medium |

### Generation Methods Summary

1. **Fully Automated (Template-based)**:
   - **ViCLEVR**: Best for scalability and systematic coverage
   - **Pros**: Scalable, consistent, balanced coverage
   - **Cons**: Less natural, limited diversity, predictable

2. **Semi-Automated (Template + Human)**:
   - **ViVQA**: Combines template structure with human refinement
   - **EVJVQA**: Domain templates with human validation
   - **Pros**: Balance of scale and naturalness
   - **Cons**: Still limited by template constraints

3. **Fully Manual (Human-Generated)**:
   - **OpenViVQA, ViTextVQA, ViOCRVQA**: Maximum naturalness
   - **Pros**: Natural language, diverse, context-aware
   - **Cons**: Expensive, slow, hard to scale

### Quality Assessment

#### Generation Model Quality (Compared to Current State-of-Art)

**Template-based approaches** (ViVQA, ViCLEVR, EVJVQA):
- **Adequate for diagnostic evaluation** but limited naturalness
- **Recommendation**: Could be enhanced with modern LLM-based generation (e.g., GPT-4, Vietnamese LLMs like PhoBERT-based generators)
- **Gap**: Modern generative models can produce more diverse and natural questions while maintaining template-like coverage

**Human-annotated approaches** (OpenViVQA, ViTextVQA, ViOCRVQA):
- **High quality** but not scalable
- **Recommendation**: Hybrid approach using LLMs for initial generation + human refinement
- **Modern alternative**: Use few-shot prompting with Vietnamese LLMs to generate candidate questions, then human validation

---

## Recommendations and Contributions

### Identified Gaps and Improvement Opportunities

#### 1. Template-Based Datasets Need Enhancement

**ViVQA Improvements**:
- ✅ Expand question type templates beyond current set
- ✅ Add complex reasoning templates (spatial, temporal, causal)
- ✅ Include more Vietnamese-specific linguistic patterns
- ✅ Add templates for:
  - Comparative questions ("Which is larger, X or Y?")
  - Counting with conditions ("How many red objects?")
  - Spatial relationships ("What is to the left of...?")
  - Attribute combinations ("What color is the largest object?")

**ViCLEVR Improvements** (if/when integrated):
- ✅ Ensure templates produce natural Vietnamese
- ✅ Add Vietnamese spatial terms and relational expressions
- ✅ Validate templates with native speakers
- ✅ Expand compositional complexity in templates

**EVJVQA Improvements**:
- ✅ Develop comprehensive Vietnamese e-commerce templates
- ✅ Add comparison and recommendation templates
- ✅ Include Vietnamese shopping context (sizes, measurements, terms)

#### 2. Missing Dataset Types

**Identified Missing Coverage**:
- ✅ **ViCLEVR**: Not yet in repository - should be added
  - Would provide systematic reasoning evaluation
  - Complements real-image datasets
  - Enables diagnostic analysis of model capabilities

#### 3. Generation Method Modernization

**Current State vs. Modern Approaches**:

**What datasets use currently**:
- Manual templates (fixed patterns)
- Human annotation (expensive, slow)
- Translation from English (may not be natural)

**Modern approaches to consider**:
- 🔄 **LLM-based generation**: Use Vietnamese language models (PhoBERT, ViT5, etc.) to generate questions
- 🔄 **Few-shot prompting**: Use GPT-4 or Claude with Vietnamese examples to generate diverse questions
- 🔄 **Back-translation augmentation**: Generate variations using translation cycles
- 🔄 **Paraphrasing models**: Use Vietnamese paraphrasing to diversify templates
- 🔄 **Active learning**: Identify uncertain cases for human annotation

**Quality Comparison**:
- Traditional templates: ⭐⭐⭐ (adequate but limited)
- Current generation models (2024): ⭐⭐⭐⭐ (good quality with proper prompting)
- Recommended: Hybrid approach combining both ⭐⭐⭐⭐⭐

#### 4. Proposed Contributions

**High-Impact Contributions**:

1. **Template Enhancement Package**:
   - Expanded template library for Vietnamese VQA
   - Covering 10+ question types per dataset
   - Validated by native speakers
   - Integrated into this repository

2. **ViCLEVR Integration**:
   - Add ViCLEVR dataset support to repository
   - Implement data loader (similar to existing datasets)
   - Add usage example
   - Update documentation

3. **Hybrid Generation Pipeline**:
   - LLM-based question generator for Vietnamese
   - Template-guided generation for consistency
   - Human validation interface
   - Quality metrics and filtering

4. **Dataset Quality Analysis Tools**:
   - Diversity metrics (vocabulary, question types)
   - Naturalness scoring
   - Bias detection
   - Coverage analysis

5. **Annotation Guidelines**:
   - Best practices for Vietnamese VQA annotation
   - Quality control procedures
   - Common pitfalls and solutions

### Human Labeling Effort Summary

**Effort Ranking** (Highest to Lowest):
1. **OpenViVQA**: Fully manual, open-domain ⚠️⚠️⚠️⚠️⚠️ (Very High)
2. **ViTextVQA**: Manual with text complexity ⚠️⚠️⚠️⚠️ (High)
3. **ViOCRVQA**: Manual with domain focus ⚠️⚠️⚠️⚠️ (High)
4. **ViVQA**: Template-guided human annotation ⚠️⚠️⚠️ (Medium-High)
5. **EVJVQA**: Domain-specific templates ⚠️⚠️ (Medium)
6. **ViCLEVR**: Automated with minimal human validation ⚠️ (Low)

**Cost-Benefit Analysis**:
- High manual effort → High quality, low scalability
- Low manual effort → High scalability, needs quality validation
- **Optimal**: Hybrid approaches balancing both

---

## Conclusion

### Key Findings

1. **Diverse Methodologies**: Vietnamese VQA datasets employ various construction approaches, from fully automated to fully manual
2. **Template Methods Are Common**: Many datasets use templates, which can be improved with modern techniques
3. **Quality vs. Scalability Trade-off**: Manual annotation provides quality but limits scalability
4. **Generation Models Are Adequate**: Current template methods work but are not state-of-the-art compared to modern LLM-based generation
5. **Room for Improvement**: All template-based datasets can benefit from enhanced templates and modern generation approaches

### Strategic Recommendations

1. **Short-term** (Immediate):
   - Add ViCLEVR dataset support
   - Expand existing templates for ViVQA, EVJVQA
   - Document current templates and generation methods

2. **Medium-term** (3-6 months):
   - Develop LLM-based question generation pipeline
   - Create comprehensive template libraries
   - Build validation and quality metrics tools

3. **Long-term** (6-12 months):
   - Establish hybrid annotation pipeline (AI + Human)
   - Create unified framework for Vietnamese VQA dataset generation
   - Contribute enhanced datasets back to community

### Final Notes

Vietnamese VQA is a growing field with significant potential. While current datasets provide a solid foundation, there are clear opportunities for contributions through:
- Enhanced template diversity
- Modern generation methods
- Systematic quality improvements
- Better tooling and documentation

This survey provides the foundation for making informed decisions about dataset improvements and contributions to the Vietnamese VQA research community.

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-12  
**Contributors**: Dataset Survey Team
