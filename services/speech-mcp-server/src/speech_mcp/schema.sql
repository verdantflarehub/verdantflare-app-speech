CREATE TABLE IF NOT EXISTS speech_tasks (
  task_id text PRIMARY KEY,
  organization_id text NOT NULL,
  subject_id text NOT NULL,
  project_id text NOT NULL,
  scope text NOT NULL,
  idem text NOT NULL,
  input_hash text NOT NULL,
  request jsonb NOT NULL,
  principal jsonb NOT NULL,
  state jsonb NOT NULL,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','running','succeeded','failed')),
  phase text NOT NULL DEFAULT 'reference',
  last_error text,
  lease_token text,
  lease_until timestamptz,
  next_attempt timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(scope, idem)
);
CREATE INDEX IF NOT EXISTS speech_tasks_ready ON speech_tasks(next_attempt, created_at)
  WHERE status IN ('pending','running');
CREATE INDEX IF NOT EXISTS speech_tasks_gpu_held ON speech_tasks(task_id)
  WHERE state->>'gpu_held'='true' AND status!='succeeded';

CREATE TABLE IF NOT EXISTS speech_gpu_pending_releases (
  lease_id text PRIMARY KEY,
  fencing_token bigint NOT NULL,
  payload jsonb NOT NULL,
  payload_version bigint NOT NULL DEFAULT 1,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','quarantined_400')),
  owner_pod_uid text NOT NULL DEFAULT '',
  claim_token text,
  claim_until timestamptz,
  last_error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE speech_gpu_pending_releases
  ADD COLUMN IF NOT EXISTS payload_version bigint NOT NULL DEFAULT 1;
CREATE INDEX IF NOT EXISTS speech_gpu_pending_releases_ready
  ON speech_gpu_pending_releases(status, claim_until, updated_at);
