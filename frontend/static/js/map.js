/* ── EcoAlert — Harta comunitar\u0103 (Waze-style) ───────────────────────────── */

function mapApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    mapInstance: null,
    heatLayer: null,
    markersLayer: null,
    zonesLayer: null,
    mapReports: [],
    mapZones: [],
    mapLoading: false,
    mapInitialized: false,

    // View mode: 'heat' | 'markers' | 'zones'
    mapViewMode: 'zones',

    // User location
    userLat: null,
    userLng: null,
    userMarker: null,
    geoLoading: false,
    geoAvailable: !!navigator.geolocation,

    // Nearby
    nearbyReports: [],
    nearbyLoading: false,
    nearbyRadius: 1.0,

    // Filters
    mapFilterResolved: '',     // '' | '0' | '1'
    mapFilterMaterial: '',     // '' | 'plastic' | 'paper' | 'glass' | 'metal' | 'other'

    // Sidebar
    selectedZone: null,
    mapSidebarOpen: false,

    /* ── Init ─────────────────────────────────────────────────────────── */
    initMap() {
      // Refresh data when new reports come in
      window.addEventListener('eco:newReport', () => {
        if (this.mapInitialized) {
          this.loadMapData();
        }
      });
    },

    ensureMap() {
      if (this.mapInitialized) {
        // Always resize on tab switch
        setTimeout(() => { if (this.mapInstance) this.mapInstance.invalidateSize(); }, 100);
        return;
      }
      this.mapInitialized = true;

      this.$nextTick(() => {
        const container = document.getElementById('ecoMap');
        if (!container) return;

        // Init map centered on Romania
        this.mapInstance = L.map('ecoMap', {
          zoomControl: false,
          attributionControl: true,
        }).setView([44.43, 26.10], 13);

        // Add zoom control top-right
        L.control.zoom({ position: 'topright' }).addTo(this.mapInstance);

        // Custom tile layer — CartoDB dark
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
          attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
          subdomains: 'abcd',
          maxZoom: 19,
        }).addTo(this.mapInstance);

        // Layer groups
        this.markersLayer = L.layerGroup().addTo(this.mapInstance);
        this.zonesLayer = L.layerGroup().addTo(this.mapInstance);

        // Leaflet.heat
        if (typeof L.heatLayer === 'function') {
          this.heatLayer = L.heatLayer([], {
            radius: 40,
            blur: 30,
            maxZoom: 17,
            gradient: {
              0.1: '#22c55e',
              0.4: '#eab308',
              0.7: '#f97316',
              1.0: '#ef4444',
            },
          }).addTo(this.mapInstance);
          this.heatLayer.setOpacity && this.heatLayer.setOpacity(0);
        }

        // Load data + locate user
        this.loadMapData();
        this._locateUser();
      });
    },

    /* ── Data loading ───────────────────────────────────────────────────── */
    async loadMapData() {
      this.mapLoading = true;
      try {
        let reportsUrl = '/api/map/reports?limit=500';
        if (this.mapFilterResolved !== '') reportsUrl += `&resolved=${this.mapFilterResolved}`;
        if (this.mapFilterMaterial)        reportsUrl += `&material=${encodeURIComponent(this.mapFilterMaterial)}`;

        const [reports, zones] = await Promise.all([
          fetchAPI(reportsUrl),
          fetchAPI('/api/zones'),
        ]);
        this.mapReports = reports;
        this.mapZones = zones;
        this._renderMap();
      } catch (e) {
        // silently fail — map still shows
      } finally {
        this.mapLoading = false;
      }
    },

    _renderMap() {
      this._clearLayers();
      if (this.mapViewMode === 'zones') {
        this._renderZones();
      } else if (this.mapViewMode === 'markers') {
        this._renderMarkers();
      } else if (this.mapViewMode === 'heat') {
        this._renderHeat();
      }
    },

    /* ── Clear ──────────────────────────────────────────────────────────── */
    _clearLayers() {
      this.markersLayer?.clearLayers();
      this.zonesLayer?.clearLayers();
      if (this.heatLayer) this.heatLayer.setLatLngs([]);
    },

    /* ── Zone circles (primary view) ──────────────────────────────────── */
    _renderZones() {
      if (!this.zonesLayer) return;

      const sevColors = ['#22c55e', '#eab308', '#f97316', '#ef4444'];
      const sevBgColors = ['#166534', '#713f12', '#7c2d12', '#7f1d1d'];

      for (const zone of this.mapZones) {
        const color = sevColors[zone.severity] || sevColors[0];
        const bg = sevBgColors[zone.severity] || sevBgColors[0];
        const radius = 80 + zone.total_objects * 8;   // bigger = more trash

        const circle = L.circle([zone.lat, zone.lng], {
          radius: Math.min(radius, 500),
          color: color,
          fillColor: color,
          fillOpacity: 0.25,
          weight: 2,
        });

        const sev = getSeverity(zone.severity);
        const matList = Object.entries(zone.materials)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([mat, cnt]) => `<span style="background:${MATERIAL_COLORS[mat]?.light || '#f3f4f6'};color:${MATERIAL_COLORS[mat]?.text || '#374151'};padding:2px 8px;border-radius:99px;font-size:11px;margin:2px;display:inline-block">${mat} ${cnt}</span>`)
          .join('');

        const popupHtml = `
          <div style="min-width:220px;font-family:'Inter',sans-serif">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
              <span style="font-size:20px">${sev.icon}</span>
              <div>
                <div style="font-weight:700;font-size:14px;color:${color}">${sev.label}</div>
                <div style="font-size:11px;color:#9ca3af">${zone.total_reports} rapoarte · ${zone.total_objects} obiecte</div>
              </div>
            </div>
            <div style="margin:6px 0">${matList}</div>
            ${zone.last_scan ? `<div style="font-size:10px;color:#6b7280;margin-top:6px">Ultima scanare: ${timeAgo(zone.last_scan)}</div>` : ''}
          </div>
        `;

        circle.bindPopup(popupHtml, { maxWidth: 280, className: 'eco-popup' });
        circle.on('click', () => {
          this.selectedZone = zone;
          this.mapSidebarOpen = true;
        });
        circle.addTo(this.zonesLayer);
      }

      // Fit bounds to zones
      if (this.mapZones.length > 0) {
        const bounds = L.latLngBounds(this.mapZones.map(z => [z.lat, z.lng]));
        this.mapInstance.fitBounds(bounds, { padding: [60, 60], maxZoom: 15 });
      }
    },

    /* ── Individual markers ─────────────────────────────────────────────── */
    _renderMarkers() {
      if (!this.markersLayer) return;

      const sevColors = ['#22c55e', '#eab308', '#f97316', '#ef4444'];

      for (const r of this.mapReports) {
        let color = '#3b82f6'; // default
        if (r.is_resolved) {
          color = '#10b981'; // Solved -> Green
        } else {
          const sev = severityFromCount(r.total_objects);
          color = sevColors[sev];
        }

        const icon = L.divIcon({
          className: '',
          html: `<div class="eco-marker" style="background:${color};">${r.total_objects}</div>`,
          iconSize: [34, 34],
          iconAnchor: [17, 17],
        });

        const popupHtml = `
          <div style="min-width:200px;font-family:'Inter',sans-serif">
            <strong style="font-size:13px">${r.filename}</strong><br>
            <span style="color:${color};font-weight:700">${r.total_objects} obiecte</span>
            <span style="color:#9ca3af"> · ${r.inference_ms.toFixed(0)} ms</span><br>
            <small style="color:#6b7280">${timeAgo(r.upload_time)}</small>
            ${r.address ? `<div style="font-size:11px;color:#6b7280;margin-top:4px;line-height:1.2;white-space:normal">📍 ${r.address}</div>` : ''}
            ${r.annotated_path ? `<br><img src="${getAnnotatedUrl(r.annotated_path)}" style="width:100%;margin-top:6px;border-radius:6px;max-height:120px;object-fit:cover" />` : ''}
          </div>
        `;

        L.marker([r.latitude, r.longitude], { icon })
          .bindPopup(popupHtml, { maxWidth: 280, className: 'eco-popup' })
          .addTo(this.markersLayer);
      }

      if (this.mapReports.length > 0) {
        const bounds = L.latLngBounds(this.mapReports.map(r => [r.latitude, r.longitude]));
        this.mapInstance.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
      }
    },

    /* ── Heat layer ─────────────────────────────────────────────────────── */
    _renderHeat() {
      if (!this.heatLayer) return;
      const points = this.mapReports.map(r => [r.latitude, r.longitude, r.total_objects]);
      this.heatLayer.setLatLngs(points);
    },

    /* ── View mode switch ───────────────────────────────────────────────── */
    setViewMode(mode) {
      this.mapViewMode = mode;
      this._renderMap();
    },

    /* ── User location ──────────────────────────────────────────────────── */
    _locateUser() {
      if (!navigator.geolocation) return;
      this.geoLoading = true;
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          this.userLat = pos.coords.latitude;
          this.userLng = pos.coords.longitude;
          this.geoLoading = false;
          this._addUserMarker();
          if (this.mapZones.length === 0 && this.mapReports.length === 0) {
            this.mapInstance.setView([this.userLat, this.userLng], 15);
          }
        },
        () => { this.geoLoading = false; },
        { enableHighAccuracy: true, timeout: 12000 }
      );
    },

    _addUserMarker() {
      if (!this.mapInstance || !this.userLat) return;
      if (this.userMarker) this.userMarker.remove();
      this.userMarker = L.marker([this.userLat, this.userLng], {
        icon: L.divIcon({
          className: '',
          html: `<div class="eco-user-marker"></div>`,
          iconSize: [20, 20],
          iconAnchor: [10, 10],
        }),
        zIndexOffset: 1000,
      })
        .addTo(this.mapInstance)
        .bindPopup('<div style="font-family:Inter,sans-serif;font-size:12px">📍 Locația ta</div>');
    },

    centerOnUser() {
      if (this.userLat && this.mapInstance) {
        this.mapInstance.setView([this.userLat, this.userLng], 15, { animate: true });
      } else {
        this._locateUser();
      }
    },

    /* ── Nearby ─────────────────────────────────────────────────────────── */
    async loadNearby() {
      if (!this.userLat) {
        showToast('Activează GPS-ul mai întâi', 'error');
        return;
      }
      this.nearbyLoading = true;
      try {
        this.nearbyReports = await fetchAPI(
          `/api/nearby?lat=${this.userLat}&lng=${this.userLng}&radius_km=${this.nearbyRadius}&limit=20`
        );
      } catch (e) {
        showToast('Eroare la căutarea rapoartelor apropiate', 'error');
      } finally {
        this.nearbyLoading = false;
      }
    },

    /* ── Stats ──────────────────────────────────────────────────────────── */
    get mapStats() {
      const total = this.mapReports.length;
      const objects = this.mapReports.reduce((s, r) => s + r.total_objects, 0);
      const highZones = this.mapZones.filter(z => z.severity >= 2).length;
      return { total, objects, highZones, zones: this.mapZones.length };
    },

    get selectedZoneSeverity() {
      if (!this.selectedZone) return null;
      return getSeverity(this.selectedZone.severity);
    },

    getSeverity(level) { return getSeverity(level); },
    timeAgo(iso) { return timeAgo(iso); },
    formatDate(iso) { return formatDate(iso); },
    materialBadgeStyle(m) { return materialBadgeStyle(m); },
    materialEntries(materialsObj) {
      return Object.entries(materialsObj || {}).sort((a, b) => b[1] - a[1]);
    },
  };
}
