# Vietnamese VQA Datasets - Quick Reference Guide

This document provides a quick reference summary of Vietnamese VQA datasets analyzed in our comprehensive survey.

## Quick Comparison Table

| Dataset | Domain | Images | Q&A Method | Scalability | Human Effort | Best For |
|---------|--------|--------|------------|-------------|--------------|----------|
| **ViVQA** | General | MS COCO | Template + Human | ⭐⭐⭐ | ⚠️⚠️⚠️ | Baseline VQA research |
| **OpenViVQA** | Open-domain | Web-crawled | Fully Human | ⭐ | ⚠️⚠️⚠️⚠️⚠️ | Natural language VQA |
| **ViTextVQA** | Scene Text | Text images | Human | ⭐ | ⚠️⚠️⚠️⚠️ | Text reading in images |
| **ViOCRVQA** | OCR/Documents | Documents | Human | ⭐⭐ | ⚠️⚠️⚠️⚠️ | Document understanding |
| **ViCLEVR** | Reasoning | Synthetic 3D | Automated | ⭐⭐⭐⭐⭐ | ⚠️ | Diagnostic evaluation |
| **EVJVQA** | E-commerce | Products | Domain Templates | ⭐⭐⭐⭐ | ⚠️⚠️ | Shopping applications |

*Scalability: ⭐ = Low, ⭐⭐⭐⭐⭐ = Very High*  
*Human Effort: ⚠️ = Low, ⚠️⚠️⚠️⚠️⚠️ = Very High*

## Dataset Characteristics

### ViVQA
- **Papers**: First Vietnamese VQA dataset
- **Size**: Train + Test splits
- **Image Source**: MS COCO
- **Question Types**: Object recognition, basic reasoning
- **Template Method**: ✅ Uses templates
- **Improvement Areas**: Expand templates, add complex reasoning

### OpenViVQA
- **Papers**: Open-domain Vietnamese VQA
- **Size**: Train + Dev + Test splits
- **Image Source**: Web-crawled Vietnamese content
- **Question Types**: Diverse open-domain questions
- **Template Method**: ❌ Minimal templates, mostly human-generated
- **Improvement Areas**: Semi-automatic generation for scaling

### ViTextVQA
- **Papers**: Text-based VQA for Vietnamese
- **Size**: Train + Dev + Test splits
- **Image Source**: Images with Vietnamese text
- **Question Types**: Text reading, comprehension
- **Template Method**: 🔶 Hybrid approach
- **Improvement Areas**: Text reasoning templates

### ViOCRVQA
- **Papers**: OCR-focused VQA
- **Size**: Train + Dev + Test splits
- **Image Source**: Documents, books, products
- **Question Types**: Text extraction, comprehension
- **Template Method**: 🔶 Template-friendly domain
- **Improvement Areas**: Multi-step text reasoning

### ViCLEVR
- **Papers**: Vietnamese CLEVR (expected)
- **Size**: TBD (not yet available)
- **Image Source**: Synthetic 3D scenes
- **Question Types**: Counting, comparison, spatial reasoning
- **Template Method**: ✅ Pure template-based
- **Improvement Areas**: Natural Vietnamese templates

### EVJVQA
- **Papers**: E-commerce multilingual VQA
- **Size**: Train split
- **Image Source**: E-commerce products
- **Question Types**: Product attributes, features
- **Template Method**: ✅ Domain-specific templates
- **Improvement Areas**: Vietnamese shopping context

## Generation Methods Summary

### Template-Based (Good Enough? Assessment)

**ViVQA Templates**: ⭐⭐⭐ Adequate
- Current: Basic templates for object questions
- Modern: Could use LLM-enhanced generation
- Verdict: **Needs improvement** - Limited diversity

**ViCLEVR Templates**: ⭐⭐⭐⭐ Good
- Current: Systematic compositional templates
- Modern: On par with state-of-art for diagnostic tasks
- Verdict: **Good enough** for intended purpose

**EVJVQA Templates**: ⭐⭐⭐ Adequate
- Current: E-commerce domain templates
- Modern: Could expand with Vietnamese shopping terms
- Verdict: **Needs improvement** - Missing Vietnamese context

### Human-Annotated (Quality vs. Cost)

**OpenViVQA**: ⭐⭐⭐⭐⭐ Excellent quality, very high cost
- Best for: Natural language patterns
- Challenge: Not scalable
- Recommendation: Hybrid approach (LLM + human validation)

**ViTextVQA**: ⭐⭐⭐⭐ High quality, high cost
- Best for: Text understanding
- Challenge: Complex annotation process
- Recommendation: Template-assist for common patterns

**ViOCRVQA**: ⭐⭐⭐⭐ High quality, high cost
- Best for: OCR tasks
- Challenge: Text verification overhead
- Recommendation: Semi-automatic OCR + human validation

## Contribution Opportunities

### High Priority
1. **Expand ViVQA Templates** - Add 10+ new template types
2. **Add ViCLEVR Support** - Integrate dataset when available
3. **Vietnamese E-commerce Templates for EVJVQA** - 20+ shopping context templates

### Medium Priority
4. **Template Enhancement Library** - Centralized template management
5. **LLM-based Generation Pipeline** - Modern question generation
6. **Quality Metrics Tools** - Automated quality assessment

### Low Priority (Long-term)
7. **Hybrid Annotation System** - AI + human pipeline
8. **Cross-dataset Analysis** - Unified evaluation framework

## Template Improvements Checklist

### For ViVQA
- [ ] Spatial reasoning templates (10+ types)
- [ ] Counting with conditions (5+ types)
- [ ] Comparative questions (5+ types)
- [ ] Attribute combinations (10+ types)
- [ ] Causal reasoning templates (5+ types)

### For ViCLEVR (when available)
- [ ] Vietnamese spatial terms validation
- [ ] Natural Vietnamese question patterns
- [ ] Compositional template expansion
- [ ] Native speaker review

### For EVJVQA
- [ ] Vietnamese shopping terms (20+ terms)
- [ ] Product comparison templates (10+ types)
- [ ] Size/measurement templates (Vietnamese standards)
- [ ] Recommendation question templates (5+ types)

## Key Findings

### What Works Well
✅ **Template methods** for diagnostic tasks (ViCLEVR)  
✅ **Human annotation** for natural language (OpenViVQA)  
✅ **Domain-specific templates** for e-commerce (EVJVQA)  
✅ **Hybrid approaches** for text tasks (ViTextVQA, ViOCRVQA)

### What Needs Improvement
❌ Limited template diversity in ViVQA  
❌ Scalability issues in human-annotated datasets  
❌ Missing Vietnamese-specific linguistic patterns  
❌ No modern LLM-based generation methods  
❌ ViCLEVR not yet integrated

### Modern vs. Current Generation

**Current (2020-2023)**:
- Manual templates: Limited but structured
- Human annotation: High quality but expensive
- Translation: Often unnatural

**Modern (2024+)**:
- LLM generation: High diversity, good quality
- Few-shot prompting: Efficient with validation
- Hybrid AI+Human: Best of both worlds

**Verdict**: Current methods are **adequate but not optimal**. Modern LLM-based approaches would significantly improve quality and scalability.

## References

For detailed analysis, see:
- [Dataset Implementations](../dataset/)
- [Usage Examples](../examples/)

---

**Last Updated**: 2025-11-12  
**Version**: 1.0
