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
