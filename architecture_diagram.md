# PDF Ingestor Pipeline - Technical Architecture

This document provides a highly detailed architectural diagram of the PDF Ingestor pipeline, tracking a document from upload through final database persistence.

```mermaid
flowchart TD
    %% Global Styling
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#000;
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef azure fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef db fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef logic fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000;

    %% Entry Point
    InputPDF([Input PDF Document]):::input
    
    subgraph "Phase 0: Preprocessing & Deduplication"
        HashDoc[Compute SHA-256 Hash]:::logic
        CheckDB{Hash exists in DB?}:::logic
        SkipProcess([Skip - Duplicate Found])
        Uploader[uploader.py: Validate & Load PDF]:::logic
        InitDB[(PostgreSQL: Insert Processing Record)]:::db
        
        InputPDF --> HashDoc
        HashDoc --> CheckDB
        CheckDB -->|Yes| SkipProcess
        CheckDB -->|No| Uploader
        Uploader --> InitDB
    end

    subgraph "Phase 1: Classification & Parallel Page Processing"
        Classifier[classifier.py: Flag 'text', 'scanned', 'text_with_images']:::logic
        ThreadPool[[ThreadPoolExecutor - Process Pages Concurrently]]:::logic
        
        Uploader --> Classifier
        Classifier --> ThreadPool
        
        ThreadPool --> TextRouter{Is Scanned or Image?}
        TextRouter -->|No| NativeText[text_extractor.py: pdfplumber]
        TextRouter -->|Yes| AzureVision[ocr_extractor.py: Azure Vision OCR]:::azure
        
        NativeText --> RawText(Raw Text)
        AzureVision --> RawText
        
        RawText -.-> TableExt[table_extractor.py: Extract Tabular Data via Azure Vision]:::azure
        TableExt --> ExtTables(Extracted Tables)
        
        RawText -.-> CheckboxExt[checkbox_extractor.py: Detect Checkbox Geometries]
        CheckboxExt --> RawCheckboxes(Raw Checkboxes)
    end

    subgraph "Phase 2: Template Routing & Generation"
        AggrText(Aggregated Full Document Text)
        RawText --> AggrText
        
        TempRouter{Match Built-in Static Templates?}:::logic
        AggrText --> TempRouter
        
        TempRouter -->|Yes| StaticExec[template_matcher.py: Apply Static Regex Patterns]
        
        TempRouter -->|No| CacheCheck{Fingerprint Match in Cached JSON Templates?}:::logic
        CacheCheck -->|Yes| LoadCache[Load Cached JSON Template]
        
        CacheCheck -->|No| LLMGen[llm_template_generator.py: Request Dynamic Template]:::azure
        LLMGen --> SaveCache[Save JSON to generated_templates/]
        SaveCache --> LoadCache
        
        LoadCache --> DynamicExec[template_matcher.py: Apply Auto-Anchored Regex]
    end

    subgraph "Phase 3: Data Extraction & Resolution"
        StaticExec --> RawKV(Raw Key-Value Pairs)
        DynamicExec --> RawKV
        
        LLMGroups(LLM Checkbox Groupings)
        LoadCache -.-> LLMGroups
        
        ResolveElections[election_resolver.py: Map Checkboxes to Groups]:::logic
        RawCheckboxes --> ResolveElections
        LLMGroups --> ResolveElections
        ResolveElections --> ResolvedKV(Resolved Election KVs)
    end

    subgraph "Phase 4: Validation & Quality Control"
        CombineKV(Combined Key-Values)
        RawKV --> CombineKV
        ResolvedKV --> CombineKV
        
        FieldVal[field_validators.py: Clean TINs, Amounts, Dates]:::logic
        ExtrVal[extraction_validator.py: Check Logical Consistencies]:::logic
        QualityGate[Compute Quality Score %]:::logic
        
        CombineKV --> FieldVal
        FieldVal --> ExtrVal
        ExtrVal --> QualityGate
        QualityGate --> FinalKV(Validated Key-Values)
    end

    subgraph "Phase 5: Persistence & Cloud Archival"
        UpdateDB[(PostgreSQL: Update Record with KVs, Tables, Quality Score, Status)]:::db
        FinalKV --> UpdateDB
        ExtTables --> UpdateDB
        RawCheckboxes --> UpdateDB
        
        UploadBlob[blob_uploader.py: Archive to Azure Storage]:::azure
        AggrText --> UploadBlob
        InputPDF -.-> UploadBlob
        UploadBlob --> AzureBlobStorage[(Azure Blob: raw_files/ & raw_txt_files/)]:::db
    end

```

### Diagram Highlights
1. **Preprocessing**: The system uses `hashlib.sha256` to short-circuit processing if the exact same file was already processed.
2. **Parallel Extraction**: The `ThreadPoolExecutor` dispatches pages concurrently to Azure Vision (for OCR/tables) and native extractors.
3. **Template Routing**: It favors fast local execution (Static Templates -> Cached Fingerprints) before calling out to Azure OpenAI for dynamic regex generation.
4. **Resolution**: Checkboxes identified visually in Phase 1 are mapped to semantic meanings defined by the LLM in Phase 2.
5. **Quality Control**: Data is validated and scored before final insertion into PostgreSQL.
