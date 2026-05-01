# telecom
# BOSS Map API

BOSS Map API is a FastAPI-based backend service for processing geospatial network infrastructure data and preparing it for visualization on a frontend map.

The service reads raw data about RK/KYA devices, subscriber addresses, cabinets, roads, rivers, and manual corrections. It then matches addresses with the most suitable network connection points and returns structured GeoJSON-like data that can be used by a map interface.

## Overview

The main purpose of this project is to automate the preparation of infrastructure map data for BOSS.

The API performs the following tasks:

- reads RK/KYA and cabinet data from CSV files;
- reads subscriber address data from CSV files;
- reads road and river geometry from a GeoJSON file;
- applies manual corrections from `changes.json`, if available;
- normalizes and compares address data;
- assigns subscriber addresses to suitable RK/KYA devices;
- separates internal and external connection logic;
- checks distance limits and available ports;
- prevents invalid external links that cross rivers or roads incorrectly;
- groups RK devices located at the same coordinates;
- returns processed data through API endpoints.

The project is designed to be used together with a frontend map application.

## Tech Stack

- Python
- FastAPI
- Pandas
- NumPy
- SciPy
- Shapely
- Uvicorn

## Project Structure

```text
.
├── main.py
├── paths.config.json
├── changes.json
├── copper_distrib.csv
├── addresses.csv
├── roads_rivers.geojson
└── README.md
