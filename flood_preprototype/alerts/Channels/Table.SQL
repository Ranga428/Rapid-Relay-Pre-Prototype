create table if not exists flood_predictions (
  id uuid primary key default uuid_generate_v4(),
  timestamp timestamptz unique,
  flood_probability float,
  risk_tier text,
  created_at timestamp default now()
);

create table public.obando_environmental_data (
    id         bigint generated always as identity primary key,
    timestamp  timestamptz  not null,
    distance_m numeric      not null,
    soil       numeric      not null,
    humidity   numeric      not null
);