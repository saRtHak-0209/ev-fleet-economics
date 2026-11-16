-- VoltMove Mobility — operational & financial store (spec §Module 1)
-- PostgreSQL-compatible DDL. SQLAlchemy creates the same structure on
-- SQLite automatically; this file is the reference schema.

CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id        VARCHAR(16) PRIMARY KEY,
    city              VARCHAR(32)      NOT NULL,
    cohort            VARCHAR(16)      NOT NULL,     -- main | pilot
    purchase_cost     NUMERIC(12, 2)   NOT NULL,
    battery_capacity  NUMERIC(6, 2)    NOT NULL,     -- kWh
    rated_range       NUMERIC(6, 1)    NOT NULL,     -- km
    actual_range      NUMERIC(6, 1)    NOT NULL,     -- km observed
    purchase_date     DATE             NOT NULL,
    useful_life       INTEGER          NOT NULL,     -- months
    residual_value    NUMERIC(12, 2)   NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_operations (
    date               DATE            NOT NULL,
    vehicle_id         VARCHAR(16)     NOT NULL REFERENCES vehicles(vehicle_id),
    city               VARCHAR(32)     NOT NULL,
    vehicle_age_months NUMERIC(6, 2)   NOT NULL,
    total_km           NUMERIC(7, 2)   NOT NULL CHECK (total_km >= 0),
    productive_km      NUMERIC(7, 2)   NOT NULL CHECK (productive_km >= 0),
    availability       NUMERIC(4, 3)   NOT NULL CHECK (availability BETWEEN 0 AND 1),
    downtime_hours     NUMERIC(5, 2)   NOT NULL CHECK (downtime_hours >= 0),
    energy_consumed_kwh NUMERIC(7, 3)  NOT NULL CHECK (energy_consumed_kwh >= 0),
    charging_cost      NUMERIC(9, 2)   NOT NULL CHECK (charging_cost >= 0),
    maintenance_cost   NUMERIC(9, 2)   NOT NULL CHECK (maintenance_cost >= 0),
    battery_soh        NUMERIC(4, 3)   NOT NULL CHECK (battery_soh BETWEEN 0 AND 1),
    revenue            NUMERIC(9, 2)   NOT NULL CHECK (revenue >= 0),
    PRIMARY KEY (date, vehicle_id),
    CHECK (productive_km <= total_km)
);
CREATE INDEX IF NOT EXISTS idx_ops_vehicle ON daily_operations (vehicle_id);
CREATE INDEX IF NOT EXISTS idx_ops_city_date ON daily_operations (city, date);

-- Financial & market reference data (spec §Module 1C/1D): assumption
-- register snapshots, kept so every model run is reproducible.
CREATE TABLE IF NOT EXISTS assumptions_register (
    snapshot_date  DATE          NOT NULL,
    grp            VARCHAR(32)   NOT NULL,
    assumption     VARCHAR(64)   NOT NULL,
    value          VARCHAR(64)   NOT NULL,
    unit           VARCHAR(128)  NOT NULL,
    source         VARCHAR(256)  NOT NULL,
    as_of          VARCHAR(16)   NOT NULL,
    confidence     VARCHAR(16)   NOT NULL,
    PRIMARY KEY (snapshot_date, grp, assumption)
);

CREATE TABLE IF NOT EXISTS market_reference (
    city        VARCHAR(32)  NOT NULL,
    parameter   VARCHAR(64)  NOT NULL,
    value       VARCHAR(64)  NOT NULL,
    source      VARCHAR(256) NOT NULL,
    confidence  VARCHAR(16)  NOT NULL,
    PRIMARY KEY (city, parameter)
);
