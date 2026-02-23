CREATE TABLE documents (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  tenant_id BIGINT UNSIGNED NOT NULL DEFAULT 0,
  doc_type VARCHAR(32) NOT NULL DEFAULT 'annual_report',   -- annual_report / prospectus / notice ...
  source VARCHAR(64) NOT NULL DEFAULT 'cninfo',            -- cninfo / upload / crawler ...
  title VARCHAR(512) NOT NULL,
  company_name VARCHAR(128) NULL,
  stock_code VARCHAR(32) NULL,
  year INT NULL,
  report_date DATE NULL,

  file_uri VARCHAR(1024) NOT NULL,      -- 本地路径/OSS/MinIO URL
  file_sha256 CHAR(64) NOT NULL,        -- 文件级 hash
  file_size BIGINT UNSIGNED NULL,

  parse_status VARCHAR(32) NOT NULL DEFAULT 'pending',     -- pending/parsed/failed
  parse_error TEXT NULL,

  current_version_id BIGINT UNSIGNED NULL, -- 指向最新版本（doc_versions.id）

  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

  UNIQUE KEY uk_tenant_filehash (tenant_id, file_sha256),
  KEY idx_company_year (tenant_id, stock_code, year),
  KEY idx_title (title(191))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE doc_versions (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  tenant_id BIGINT UNSIGNED NOT NULL DEFAULT 0,
  document_id BIGINT UNSIGNED NOT NULL,

  version_no INT NOT NULL,                -- 1,2,3...
  version_tag VARCHAR(64) NULL,            -- v1 / revise / 2025-xx-xx
  content_sha256 CHAR(64) NOT NULL,        -- 解析后“可检索内容”的整体 hash（可选）

  status VARCHAR(32) NOT NULL DEFAULT 'active',  -- active/archived
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

  UNIQUE KEY uk_doc_ver (document_id, version_no),
  KEY idx_doc (document_id),
  CONSTRAINT fk_ver_doc FOREIGN KEY (document_id) REFERENCES documents(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE chunks (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  tenant_id BIGINT UNSIGNED NOT NULL DEFAULT 0,
  document_id BIGINT UNSIGNED NOT NULL,
  version_id BIGINT UNSIGNED NOT NULL,

  chunk_no INT NOT NULL,                       -- 文档内序号
  page_start INT NULL,
  page_end INT NULL,
  section_path VARCHAR(512) NULL,              -- 例如 "第三节 财务信息/合并利润表"
  content LONGTEXT NOT NULL,
  content_sha256 CHAR(64) NOT NULL,            -- chunk 级 hash（增量对比）

  content_tokens INT NULL,                     -- 方便统计成本
  is_table TINYINT(1) NOT NULL DEFAULT 0,       -- 表格 chunk 标记
  table_id BIGINT UNSIGNED NULL,               -- 如果来自表格，指向 tables.id

  is_deleted TINYINT(1) NOT NULL DEFAULT 0,     -- 软删除（文档更新或重建）
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

  UNIQUE KEY uk_ver_chunkno (version_id, chunk_no),
  KEY idx_doc_ver (document_id, version_id),
  KEY idx_chunk_hash (tenant_id, content_sha256),
  FULLTEXT KEY ft_content (content),
  CONSTRAINT fk_chunk_doc FOREIGN KEY (document_id) REFERENCES documents(id),
  CONSTRAINT fk_chunk_ver FOREIGN KEY (version_id) REFERENCES doc_versions(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;




CREATE TABLE tables (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  tenant_id BIGINT UNSIGNED NOT NULL DEFAULT 0,
  document_id BIGINT UNSIGNED NOT NULL,
  version_id BIGINT UNSIGNED NOT NULL,

  page_no INT NULL,
  table_title VARCHAR(512) NULL,
  schema_json JSON NULL,              -- 列名、对齐、合并单元格等结构信息
  data_json JSON NOT NULL,            -- 表格数据（行列）
  table_sha256 CHAR(64) NOT NULL,     -- 表格级 hash（可用于增量）

  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

  KEY idx_doc_ver (document_id, version_id),
  KEY idx_table_hash (tenant_id, table_sha256),
  CONSTRAINT fk_table_doc FOREIGN KEY (document_id) REFERENCES documents(id),
  CONSTRAINT fk_table_ver FOREIGN KEY (version_id) REFERENCES doc_versions(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE embeddings (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  tenant_id BIGINT UNSIGNED NOT NULL DEFAULT 0,
  chunk_id BIGINT UNSIGNED NOT NULL,

  embed_model VARCHAR(128) NOT NULL,    -- bge-m3 / text-embedding-3-large ...
  embed_dim INT NOT NULL,
  vector_store VARCHAR(32) NOT NULL,    -- faiss/qdrant/milvus/es
  vector_id VARCHAR(128) NOT NULL,      -- 向量库里的主键

  embed_sha256 CHAR(64) NOT NULL,       -- 通常=chunks.content_sha256（也可加模型名一起 hash）
  status VARCHAR(32) NOT NULL DEFAULT 'active',  -- active/deleted
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

  UNIQUE KEY uk_chunk_model (chunk_id, embed_model),
  KEY idx_vector (vector_store, vector_id),
  KEY idx_embed_hash (tenant_id, embed_sha256),
  CONSTRAINT fk_embed_chunk FOREIGN KEY (chunk_id) REFERENCES chunks(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE index_builds (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  tenant_id BIGINT UNSIGNED NOT NULL DEFAULT 0,

  index_name VARCHAR(128) NOT NULL,      -- finance_rag_idx
  vector_store VARCHAR(32) NOT NULL,
  embed_model VARCHAR(128) NOT NULL,

  params_json JSON NOT NULL,             -- chunk_size/overlap/topk/rerank_k...
  doc_count INT NOT NULL DEFAULT 0,
  chunk_count INT NOT NULL DEFAULT 0,

  status VARCHAR(32) NOT NULL DEFAULT 'running', -- running/success/failed
  error_text TEXT NULL,

  started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at DATETIME NULL,

  KEY idx_index_name (tenant_id, index_name),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE qa_requests (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  tenant_id BIGINT UNSIGNED NOT NULL DEFAULT 0,
  user_id BIGINT UNSIGNED NULL,

  query_text TEXT NOT NULL,
  query_hash CHAR(64) NOT NULL,              -- 便于去重/缓存键
  expanded_query_json JSON NULL,             -- 扩写/改写结果

  retrieval_json JSON NULL,                  -- topK、分数、RRF等（可选）
  answer_text MEDIUMTEXT NULL,
  answer_model VARCHAR(128) NULL,            -- 哪个 LLM 给的
  answer_policy VARCHAR(64) NOT NULL DEFAULT 'cite_required', -- cite_required/refuse_if_low_conf

  confidence FLOAT NULL,
  is_refused TINYINT(1) NOT NULL DEFAULT 0,  -- 证据不足拒答

  latency_ms INT NULL,
  prompt_tokens INT NULL,
  completion_tokens INT NULL,
  cost_usd DECIMAL(10,6) NULL,

  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

  KEY idx_queryhash (tenant_id, query_hash),
  KEY idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE qa_citations (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  request_id BIGINT UNSIGNED NOT NULL,
  chunk_id BIGINT UNSIGNED NOT NULL,
  rank_no INT NOT NULL,                 -- 最终引用的第几段
  quote_text TEXT NULL,                 -- 可存一个短引用（控制长度）
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

  KEY idx_req (request_id),
  CONSTRAINT fk_cite_req FOREIGN KEY (request_id) REFERENCES qa_requests(id),
  CONSTRAINT fk_cite_chunk FOREIGN KEY (chunk_id) REFERENCES chunks(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE model_events (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  tenant_id BIGINT UNSIGNED NOT NULL DEFAULT 0,
  request_id BIGINT UNSIGNED NULL,

  model_name VARCHAR(128) NOT NULL,
  event_type VARCHAR(32) NOT NULL,      -- timeout/5xx/rate_limit/parse_error/fallback
  http_status INT NULL,
  error_code VARCHAR(64) NULL,
  error_msg TEXT NULL,

  retry_count INT NOT NULL DEFAULT 0,
  fallback_to VARCHAR(128) NULL,

  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

  KEY idx_req (request_id),
  KEY idx_model_time (model_name, created_at),
  CONSTRAINT fk_evt_req FOREIGN KEY (request_id) REFERENCES qa_requests(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;