# Vietnamese VQA Documentation

This directory contains comprehensive documentation about Vietnamese Visual Question Answering (VQA) datasets, their construction methodologies, and how to contribute improvements.

## 📚 Documentation Index

### Core Documents

#### [Dataset Survey](DATASET_SURVEY.md) 📊
**Comprehensive analysis of Vietnamese VQA datasets**

A detailed survey covering all six major Vietnamese VQA datasets:
- ViVQA, OpenViVQA, ViTextVQA, ViOCRVQA, ViCLEVR, EVJVQA
- Dataset building methodologies
- Generation methods and quality assessments
- Template-based vs. human-annotated approaches
- Human labeling efforts and costs
- Comparative analysis and recommendations

**Target Audience**: Researchers, dataset creators, project maintainers

**Read this if you want to**:
- Understand how Vietnamese VQA datasets are built
- Learn about different generation methods
- Assess dataset quality and limitations
- Find opportunities for contributions

---

#### [Dataset Quick Reference](DATASET_QUICK_REFERENCE.md) ⚡
**Quick comparison and reference guide**

A concise summary of all datasets with:
- Quick comparison table
- Dataset characteristics at a glance
- Generation methods summary
- Key findings and recommendations
- Template improvement checklist

**Target Audience**: Developers, quick reference users

**Read this if you want to**:
- Quickly compare datasets
- Get an overview without deep dive
- Check specific dataset characteristics
- Find quick answers

---

#### [Template Improvements](TEMPLATE_IMPROVEMENTS.md) 🔧
**Concrete template examples and recommendations**

Detailed template improvements for each dataset:
- 35+ ViVQA template examples (spatial, counting, comparative, etc.)
- 35+ EVJVQA Vietnamese e-commerce templates
- ViCLEVR natural Vietnamese templates
- ViOCRVQA/ViTextVQA text reasoning templates
- Implementation guidelines
- Template design principles

**Target Audience**: Template contributors, dataset developers

**Read this if you want to**:
- Create new templates
- Improve existing templates
- Understand template design
- See concrete examples

---

#### [Contributing Guidelines](CONTRIBUTING.md) 🤝
**How to contribute to Vietnamese VQA datasets**

Complete contribution guide covering:
- Priority contribution areas
- Step-by-step contribution process
- Quality standards and validation
- Contribution types (templates, code, docs, tools)
- Workflow and best practices
- Examples of good contributions

**Target Audience**: Contributors, collaborators

**Read this if you want to**:
- Contribute to the project
- Understand contribution standards
- Learn the submission process
- See contribution examples

---

## 🎯 Quick Navigation

### By Goal

**I want to understand the datasets**:
1. Start with [Quick Reference](DATASET_QUICK_REFERENCE.md)
2. Deep dive with [Dataset Survey](DATASET_SURVEY.md)

**I want to improve templates**:
1. Read [Template Improvements](TEMPLATE_IMPROVEMENTS.md)
2. Check [Contributing Guidelines](CONTRIBUTING.md)
3. Submit your contribution

**I want to contribute code**:
1. Read [Contributing Guidelines](CONTRIBUTING.md)
2. Review [Dataset Survey](DATASET_SURVEY.md) for context
3. Follow contribution workflow

**I want to add a new dataset**:
1. Study existing datasets in [Dataset Survey](DATASET_SURVEY.md)
2. Review implementation in [Template Improvements](TEMPLATE_IMPROVEMENTS.md)
3. Follow [Contributing Guidelines](CONTRIBUTING.md)

### By Dataset

**ViVQA**:
- Survey: [DATASET_SURVEY.md#vivqa](DATASET_SURVEY.md#vivqa)
- Templates: [TEMPLATE_IMPROVEMENTS.md#vivqa-template-enhancements](TEMPLATE_IMPROVEMENTS.md#vivqa-template-enhancements)
- Quick Ref: [DATASET_QUICK_REFERENCE.md](DATASET_QUICK_REFERENCE.md)

**OpenViVQA**:
- Survey: [DATASET_SURVEY.md#openvivqa](DATASET_SURVEY.md#openvivqa)
- Quick Ref: [DATASET_QUICK_REFERENCE.md](DATASET_QUICK_REFERENCE.md)

**ViTextVQA**:
- Survey: [DATASET_SURVEY.md#vitextvqa](DATASET_SURVEY.md#vitextvqa)
- Templates: [TEMPLATE_IMPROVEMENTS.md#viocrvqavitextvqa-template-extensions](TEMPLATE_IMPROVEMENTS.md#viocrvqavitextvqa-template-extensions)
- Quick Ref: [DATASET_QUICK_REFERENCE.md](DATASET_QUICK_REFERENCE.md)

**ViOCRVQA**:
- Survey: [DATASET_SURVEY.md#viocrvqa](DATASET_SURVEY.md#viocrvqa)
- Templates: [TEMPLATE_IMPROVEMENTS.md#viocrvqavitextvqa-template-extensions](TEMPLATE_IMPROVEMENTS.md#viocrvqavitextvqa-template-extensions)
- Quick Ref: [DATASET_QUICK_REFERENCE.md](DATASET_QUICK_REFERENCE.md)

**ViCLEVR**:
- Survey: [DATASET_SURVEY.md#viclevr](DATASET_SURVEY.md#viclevr)
- Templates: [TEMPLATE_IMPROVEMENTS.md#viclevr-natural-vietnamese-templates](TEMPLATE_IMPROVEMENTS.md#viclevr-natural-vietnamese-templates)
- Quick Ref: [DATASET_QUICK_REFERENCE.md](DATASET_QUICK_REFERENCE.md)

**EVJVQA**:
- Survey: [DATASET_SURVEY.md#evjvqa](DATASET_SURVEY.md#evjvqa)
- Templates: [TEMPLATE_IMPROVEMENTS.md#evjvqa-vietnamese-e-commerce-templates](TEMPLATE_IMPROVEMENTS.md#evjvqa-vietnamese-e-commerce-templates)
- Quick Ref: [DATASET_QUICK_REFERENCE.md](DATASET_QUICK_REFERENCE.md)

---

## 📈 Document Statistics

| Document | Lines | Words | Focus |
|----------|-------|-------|-------|
| Dataset Survey | ~850 | ~12,000 | Comprehensive analysis |
| Quick Reference | ~350 | ~4,000 | Quick lookup |
| Template Improvements | ~900 | ~13,000 | Practical examples |
| Contributing Guidelines | ~650 | ~9,500 | Contribution process |

**Total Documentation**: ~2,750 lines, ~38,500 words

---

## 🔑 Key Findings Summary

### Dataset Building Methods

1. **Template-based** (ViVQA, ViCLEVR, EVJVQA):
   - Good for scalability
   - Need diversity improvements
   - Can be enhanced with modern LLMs

2. **Human-annotated** (OpenViVQA, ViTextVQA, ViOCRVQA):
   - High quality
   - Expensive and slow
   - Would benefit from hybrid approaches

3. **Recommended**: Hybrid AI + Human approach

### Quality Assessment

**Current Generation Methods**:
- Template methods: ⭐⭐⭐ (adequate but limited)
- Modern LLM-based: ⭐⭐⭐⭐ (good with proper prompting)
- Hybrid approach: ⭐⭐⭐⭐⭐ (recommended)

### Top Contribution Opportunities

1. **ViVQA Template Expansion** - Add 35+ new templates
2. **EVJVQA Vietnamese Templates** - Create e-commerce templates
3. **ViCLEVR Integration** - Add dataset support
4. **Modern Generation Pipeline** - LLM-based question generation
5. **Quality Metrics Tools** - Automated quality assessment

---

## 🚀 Getting Started

### For Researchers
1. Read the [Dataset Survey](DATASET_SURVEY.md)
2. Check [Quick Reference](DATASET_QUICK_REFERENCE.md) for specific datasets
3. Cite this work if you use these insights

### For Contributors
1. Start with [Contributing Guidelines](CONTRIBUTING.md)
2. Review [Template Improvements](TEMPLATE_IMPROVEMENTS.md) for examples
3. Pick a contribution area and get started!

### For Developers
1. Check [Quick Reference](DATASET_QUICK_REFERENCE.md) for dataset overview
2. Review implementation details in [Dataset Survey](DATASET_SURVEY.md)
3. Follow [Contributing Guidelines](CONTRIBUTING.md) for code contributions

---

## 📝 Citation

If you use this survey or documentation in your research, please cite:

```bibtex
@misc{vqa_template_survey_2024,
  title={Vietnamese VQA Datasets: A Comprehensive Survey and Analysis},
  author={VQA Template Team},
  year={2024},
  howpublished={\url{https://github.com/ThuanNaN/VQA_Template}},
  note={Documentation for Vietnamese Visual Question Answering datasets}
}
```

---

## 🤝 Contributing to Documentation

Found an error? Have suggestions? Want to add more content?

1. Check [Contributing Guidelines](CONTRIBUTING.md)
2. Open an issue or discussion
3. Submit a pull request

All contributions are welcome! 🙏

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/ThuanNaN/VQA_Template/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ThuanNaN/VQA_Template/discussions)
- **Pull Requests**: [GitHub PRs](https://github.com/ThuanNaN/VQA_Template/pulls)

---

## 📅 Version History

- **v1.0** (2025-11-12): Initial comprehensive survey and documentation
  - Complete dataset analysis (6 datasets)
  - Template improvement recommendations (100+ templates)
  - Contribution guidelines
  - Quick reference guide

---

## 🙏 Acknowledgments

This documentation is based on:
- Analysis of existing Vietnamese VQA datasets
- Review of dataset papers and repositories
- Practical experience with dataset implementation
- Community feedback and contributions

**Contributors**: Dataset Survey Team

---

**Last Updated**: 2025-11-12  
**Version**: 1.0  
**Maintained by**: VQA Template Project
