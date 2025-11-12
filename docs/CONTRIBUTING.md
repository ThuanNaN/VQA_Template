# Contribution Guidelines for Vietnamese VQA Datasets

Based on the comprehensive dataset survey, this document outlines how to contribute to improving Vietnamese VQA datasets in this repository.

## Table of Contents
1. [Overview](#overview)
2. [Priority Contributions](#priority-contributions)
3. [How to Contribute](#how-to-contribute)
4. [Quality Standards](#quality-standards)
5. [Contribution Types](#contribution-types)

---

## Overview

The dataset survey identified several areas where contributions can significantly improve Vietnamese VQA research:

**Key Findings**:
- Template-based datasets need expansion and modernization
- Missing Vietnamese-specific linguistic patterns
- Quality vs. scalability trade-offs need addressing
- ViCLEVR dataset not yet integrated

**Contribution Philosophy**:
- **Quality over Quantity**: Better templates are more valuable than more templates
- **Natural Vietnamese**: All contributions must sound natural to native speakers
- **Systematic Coverage**: Ensure balanced coverage across question types
- **Practical Impact**: Focus on improvements that benefit real research

---

## Priority Contributions

### High Priority (Immediate Impact)

#### 1. ViVQA Template Expansion
**What**: Add 10+ new template types for spatial, comparative, and counting questions  
**Why**: Current templates are limited, restricting dataset diversity  
**Effort**: Medium (2-3 weeks)  
**Impact**: High - Improves most widely-used Vietnamese VQA dataset

**Specific Tasks**:
- [ ] Add 10 spatial reasoning templates
- [ ] Add 5 counting with conditions templates
- [ ] Add 5 comparative question templates
- [ ] Add 10 attribute combination templates
- [ ] Add 5 causal reasoning templates
- [ ] Validate all templates with native speakers

**Success Criteria**:
- Naturalness score ≥ 4.0/5.0 from native speakers
- All templates produce grammatically correct questions
- Template diversity increases by ≥50%

#### 2. EVJVQA Vietnamese Templates
**What**: Create comprehensive Vietnamese e-commerce templates  
**Why**: Current EVJVQA uses English; needs Vietnamese for local applications  
**Effort**: Medium (2-3 weeks)  
**Impact**: High - Enables Vietnamese e-commerce VQA

**Specific Tasks**:
- [ ] Create 20 product attribute templates
- [ ] Create 10 comparison templates
- [ ] Create 5 recommendation templates
- [ ] Add Vietnamese shopping terms dictionary
- [ ] Create Vietnamese size/measurement standards guide

**Success Criteria**:
- Cover 90%+ of common e-commerce questions
- Include Vietnamese-specific shopping context
- Templates validated by Vietnamese e-commerce domain experts

#### 3. ViCLEVR Integration
**What**: Add ViCLEVR dataset support to repository  
**Why**: Provides systematic reasoning evaluation missing from current datasets  
**Effort**: Medium (2-4 weeks)  
**Impact**: Medium-High - Complements existing datasets

**Specific Tasks**:
- [ ] Create ViCLEVR dataset loader class
- [ ] Implement natural Vietnamese templates
- [ ] Add usage example
- [ ] Update documentation
- [ ] Add to data/config.yaml with proper URL when available

**Success Criteria**:
- Compatible with existing dataset interface
- Templates sound natural in Vietnamese
- Full documentation and examples provided

### Medium Priority (Important but Less Urgent)

#### 4. Template Quality Enhancement Library
**What**: Centralized template management system  
**Why**: Easier to maintain, update, and contribute templates  
**Effort**: High (4-6 weeks)  
**Impact**: Medium - Infrastructure improvement

**Specific Tasks**:
- [ ] Design template library structure
- [ ] Implement template versioning
- [ ] Add template validation tools
- [ ] Create template contribution workflow
- [ ] Build template quality metrics

#### 5. LLM-based Question Generation Pipeline
**What**: Modern question generation using Vietnamese LLMs  
**Why**: Improve quality and scalability beyond manual templates  
**Effort**: High (6-8 weeks)  
**Impact**: High - Long-term modernization

**Specific Tasks**:
- [ ] Research Vietnamese LLM options (PhoBERT, ViT5, etc.)
- [ ] Design few-shot prompting templates
- [ ] Implement generation pipeline
- [ ] Add human validation interface
- [ ] Create quality filtering mechanisms

#### 6. Dataset Quality Analysis Tools
**What**: Automated tools to assess dataset quality  
**Why**: Enables systematic improvement and contribution validation  
**Effort**: Medium (3-4 weeks)  
**Impact**: Medium - Enables better contributions

**Specific Tasks**:
- [ ] Implement diversity metrics (vocabulary, question types)
- [ ] Add naturalness scoring mechanism
- [ ] Create bias detection tools
- [ ] Build coverage analysis
- [ ] Generate quality reports

### Low Priority (Nice to Have)

#### 7. Annotation Guidelines Document
**What**: Best practices for Vietnamese VQA annotation  
**Why**: Helps future contributors maintain quality  
**Effort**: Low (1 week)  
**Impact**: Low-Medium - Educational resource

#### 8. Cross-dataset Analysis Framework
**What**: Unified evaluation across all datasets  
**Why**: Better understanding of dataset characteristics  
**Effort**: Medium (3-4 weeks)  
**Impact**: Low - Research tool

---

## How to Contribute

### Step-by-Step Process

#### For Template Contributions

1. **Choose a Template Category**:
   - Review [TEMPLATE_IMPROVEMENTS.md](TEMPLATE_IMPROVEMENTS.md)
   - Select a template type that needs expansion
   - Check existing templates to avoid duplication

2. **Design Templates**:
   - Create 3-5 variations per template type
   - Ensure natural Vietnamese phrasing
   - Include examples with filled templates
   - Consider edge cases and variations

3. **Validate Templates**:
   - Get feedback from native Vietnamese speakers
   - Test template with real data if possible
   - Ensure grammatical correctness
   - Verify naturalness (score ≥ 4.0/5.0)

4. **Submit Contribution**:
   - Add templates to appropriate file in `dataset/` directory
   - Update documentation
   - Include validation results
   - Provide usage examples
   - Submit pull request with clear description

#### For Code Contributions

1. **Identify Issue**:
   - Check existing issues or create new one
   - Discuss approach with maintainers
   - Ensure contribution aligns with priorities

2. **Implement Solution**:
   - Follow existing code structure
   - Maintain compatibility with current datasets
   - Add tests if applicable
   - Document new functionality

3. **Test Thoroughly**:
   - Test with all relevant datasets
   - Verify backward compatibility
   - Check edge cases
   - Run existing tests

4. **Submit Pull Request**:
   - Clear description of changes
   - Link to related issues
   - Include usage examples
   - Address review feedback promptly

#### For Documentation Contributions

1. **Identify Gap**:
   - Review existing documentation
   - Identify missing or unclear information
   - Consider user perspective

2. **Create/Update Documentation**:
   - Follow existing documentation style
   - Include practical examples
   - Keep language clear and concise
   - Add diagrams if helpful

3. **Review and Submit**:
   - Proofread for accuracy
   - Check all links work
   - Ensure formatting is consistent
   - Submit pull request

---

## Quality Standards

### For Templates

**Mandatory Requirements**:
- ✅ Natural Vietnamese phrasing
- ✅ Grammatically correct
- ✅ Clear and unambiguous
- ✅ Appropriate for target dataset
- ✅ Includes usage examples

**Quality Metrics**:
- **Naturalness**: ≥ 4.0/5.0 (native speaker rating)
- **Diversity**: Templates should differ meaningfully from existing ones
- **Coverage**: Should address identified gaps
- **Usability**: Must be practical for question generation

**Validation Process**:
1. Self-review against checklist
2. Peer review by Vietnamese speaker
3. Test with sample data
4. Final maintainer review

### For Code

**Mandatory Requirements**:
- ✅ Follows existing code style
- ✅ Includes documentation
- ✅ Maintains backward compatibility
- ✅ Passes existing tests
- ✅ No breaking changes without discussion

**Code Quality**:
- Clean, readable code
- Appropriate comments
- Efficient implementation
- Error handling
- Type hints where applicable

### For Documentation

**Mandatory Requirements**:
- ✅ Accurate information
- ✅ Clear writing
- ✅ Practical examples
- ✅ Proper formatting
- ✅ Up-to-date references

**Documentation Quality**:
- Comprehensive coverage
- Easy to understand
- Well-organized
- Helpful examples
- Current and maintained

---

## Contribution Types

### 1. Template Contributions

**What to Contribute**:
- New question templates
- Template variations
- Template improvements
- Vietnamese language refinements

**Files to Update**:
- `docs/TEMPLATE_IMPROVEMENTS.md` (template documentation)
- Dataset-specific files in `dataset/` (implementation)
- Usage examples in `examples/` (demonstrations)

**Template Submission Format**:
```markdown
### Template Type: [Name]
**Category**: [Spatial/Counting/Comparative/etc.]
**Dataset**: [ViVQA/EVJVQA/etc.]

**Template**:
Vietnamese: "[Template with {placeholders}]"
English equivalent: "[English translation]"

**Examples**:
- "[Filled example 1]"
- "[Filled example 2]"
- "[Filled example 3]"

**Validation**:
- Naturalness score: [X.X/5.0]
- Reviewed by: [Reviewer names]
- Testing: [How tested]
```

### 2. Dataset Implementation

**What to Contribute**:
- New dataset loaders
- Dataset improvements
- Bug fixes
- Performance optimizations

**Files to Update**:
- `dataset/[dataset_name].py` (implementation)
- `data/config.yaml` (configuration)
- `examples/[dataset_name]_usage.py` (examples)
- `README.md` (documentation)

### 3. Documentation

**What to Contribute**:
- Tutorial improvements
- API documentation
- Usage examples
- Survey updates

**Files to Update**:
- `docs/*.md` (documentation files)
- `README.md` (main documentation)
- `examples/*.py` (code examples)

### 4. Tools and Utilities

**What to Contribute**:
- Quality metrics tools
- Template validation scripts
- Dataset analysis tools
- Visualization utilities

**Files to Update**:
- `utils/*.py` (utility functions)
- `scripts/*.py` (standalone tools)
- Documentation for new tools

---

## Contribution Workflow

### 1. Before Contributing

**Research**:
- Read the [Dataset Survey](DATASET_SURVEY.md)
- Review [Template Improvements](TEMPLATE_IMPROVEMENTS.md)
- Check existing issues and pull requests
- Understand current state and gaps

**Planning**:
- Choose contribution type
- Define scope clearly
- Estimate effort required
- Check with maintainers if unsure

### 2. During Development

**Best Practices**:
- Work in feature branch
- Commit frequently with clear messages
- Update documentation as you go
- Test thoroughly
- Ask for feedback early

**Communication**:
- Comment on related issues
- Ask questions if stuck
- Share progress updates
- Request early review if needed

### 3. Submitting Contribution

**Pull Request Checklist**:
- [ ] Code/templates are complete
- [ ] Documentation is updated
- [ ] Examples are provided
- [ ] Tests pass (if applicable)
- [ ] Validation completed
- [ ] PR description is clear
- [ ] Linked to related issues

**PR Description Template**:
```markdown
## Description
[Brief description of contribution]

## Type of Contribution
- [ ] Template
- [ ] Code
- [ ] Documentation
- [ ] Tools

## Related Issues
Fixes #[issue number]

## Changes Made
- [Change 1]
- [Change 2]
- ...

## Validation
- [How validated]
- [Test results]
- [Reviewer feedback]

## Checklist
- [ ] Documentation updated
- [ ] Examples provided
- [ ] Validated by native speaker (for templates)
- [ ] Tests pass
```

### 4. After Submission

**Review Process**:
- Maintainers will review within 1 week
- Address feedback promptly
- Make requested changes
- Re-request review when ready

**After Merge**:
- Celebrate! 🎉
- Watch for issues related to your contribution
- Help others build on your work
- Consider next contribution

---

## Recognition

**Contributors Will Be**:
- Listed in repository contributors
- Acknowledged in documentation
- Credited in related publications (if applicable)
- Invited to collaboration opportunities

**Top Contributors May**:
- Become maintainers
- Lead major initiatives
- Represent project at conferences
- Co-author research papers

---

## Getting Help

**Resources**:
- [Dataset Survey](DATASET_SURVEY.md) - Comprehensive analysis
- [Template Improvements](TEMPLATE_IMPROVEMENTS.md) - Template examples
- [Quick Reference](DATASET_QUICK_REFERENCE.md) - Dataset overview
- [README](../README.md) - Project documentation

**Communication**:
- GitHub Issues - Bug reports, feature requests
- GitHub Discussions - Questions, ideas
- Pull Request Comments - Code review, feedback

**Maintainer Contact**:
- Create issue for questions
- Tag maintainers in discussions
- Be patient and respectful

---

## Examples of Good Contributions

### Example 1: Template Contribution
```markdown
### New Spatial Templates for ViVQA

**Added 10 new spatial reasoning templates covering:**
- Positional queries (4 templates)
- Relative position (3 templates)
- Direction/orientation (3 templates)

**Validation:**
- Naturalness score: 4.5/5.0 (5 native speakers)
- Tested with 100 sample images
- All templates produce valid questions

**Files Changed:**
- docs/TEMPLATE_IMPROVEMENTS.md (added templates)
- dataset/vivqa.py (implementation)
- examples/vivqa_usage.py (examples)
```

### Example 2: Code Contribution
```markdown
### Add ViCLEVR Dataset Support

**Implementation:**
- Created ViCLEVRDataset class
- Added natural Vietnamese templates
- Implemented label encoder
- Added usage example

**Testing:**
- Unit tests for dataset loader
- Integration test with training pipeline
- Validated templates with 3 native speakers

**Files Changed:**
- dataset/viclevr.py (new file)
- examples/viclevr_usage.py (new file)
- data/config.yaml (added ViCLEVR config)
- README.md (updated dataset list)
```

### Example 3: Documentation Contribution
```markdown
### Vietnamese VQA Best Practices Guide

**Added:**
- Annotation guidelines for Vietnamese VQA
- Quality control procedures
- Common pitfalls and solutions
- Template design principles

**Value:**
- Helps new contributors
- Ensures consistent quality
- Reduces review time
- Educational resource

**Files Changed:**
- docs/ANNOTATION_GUIDELINES.md (new file)
- docs/DATASET_SURVEY.md (added references)
```

---

## Conclusion

Your contributions can significantly impact Vietnamese VQA research. Whether you're adding templates, implementing datasets, improving documentation, or building tools, every contribution matters.

**Start Small**:
- Fix a typo
- Add one template
- Improve an example
- Ask a question

**Grow Your Impact**:
- Tackle bigger issues
- Lead initiatives
- Mentor others
- Shape the future

**Thank you for contributing to Vietnamese VQA research!** 🙏

---

**Version**: 1.0  
**Last Updated**: 2025-11-12  
**Contact**: Create an issue or discussion on GitHub
