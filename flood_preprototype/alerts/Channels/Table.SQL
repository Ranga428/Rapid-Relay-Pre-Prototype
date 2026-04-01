create table if not exists flood_predictions (
  id uuid primary key default uuid_generate_v4(),
  timestamp timestamptz unique,
  flood_probability float,
  risk_tier text,
  created_at timestamp default now()
);
