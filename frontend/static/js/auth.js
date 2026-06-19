/* ── TrashDet — Authentication & Session ─────────────────────────────────── */

function authApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    user: null,
    token: getAuthToken(),
    isLoggedIn: false,
    
    // Auth modals
    showAuthModal: false,
    authMode: 'login', // 'login' | 'register'
    
    // Forms
    loginData: { username: '', password: '' },
    registerData: { username: '', email: '', password: '' },
    authLoading: false,

    // Password strength
    passwordRules: {
      minLength: false,
      hasUpper: false,
      hasLower: false,
      hasDigit: false,
      hasSpecial: false,
    },
    passwordScore: 0,
    showPasswordRules: false,

    /* ── Init ─────────────────────────────────────────────────────────── */
    async initAuth() {
      if (this.token) {
        await this.fetchMe();
      }
    },

    /* ── Password validation (real-time) ──────────────────────────────── */
    checkPasswordStrength(pw) {
      this.passwordRules.minLength = pw.length >= 8;
      this.passwordRules.hasUpper = /[A-Z]/.test(pw);
      this.passwordRules.hasLower = /[a-z]/.test(pw);
      this.passwordRules.hasDigit = /[0-9]/.test(pw);
      this.passwordRules.hasSpecial = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?`~]/.test(pw);

      const passed = Object.values(this.passwordRules).filter(Boolean).length;
      this.passwordScore = passed; // 0-5
      this.showPasswordRules = pw.length > 0;
    },

    getPasswordStrengthLabel() {
      if (this.passwordScore <= 1) return { text: 'Foarte slabă', color: '#ef4444' };
      if (this.passwordScore <= 2) return { text: 'Slabă', color: '#f97316' };
      if (this.passwordScore <= 3) return { text: 'Medie', color: '#eab308' };
      if (this.passwordScore <= 4) return { text: 'Bună', color: '#22c55e' };
      return { text: 'Puternică', color: '#16a34a' };
    },

    /* ── Login ────────────────────────────────────────────────────────── */
    async login() {
      this.authLoading = true;
      try {
        const formData = new URLSearchParams();
        formData.append('username', this.loginData.username);
        formData.append('password', this.loginData.password);

        const data = await fetchAPI('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: formData
        });

        if (data.access_token) {
          this.token = data.access_token;
          setAuthToken(this.token);
          await this.fetchMe();
          this.showAuthModal = false;
          showToast(`Bine ai revenit, ${this.user.username}!`);
          window.dispatchEvent(new CustomEvent('eco:authChanged'));
        } else {
          throw new Error('Răspuns invalid de la server');
        }
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.authLoading = false;
      }
    },

    /* ── Register ─────────────────────────────────────────────────────── */
    async register() {
      // Client-side password check
      this.checkPasswordStrength(this.registerData.password);
      if (this.passwordScore < 5) {
        showToast('Parola nu îndeplinește toate cerințele', 'error');
        return;
      }

      this.authLoading = true;
      try {
        await fetchAPI('/api/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.registerData)
        });

        showToast('Cont creat cu succes! Te poți autentifica acum.');
        this.authMode = 'login';
        this.loginData.username = this.registerData.username;
        this.showPasswordRules = false;
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.authLoading = false;
      }
    },

    /* ── Fetch current user ───────────────────────────────────────────── */
    async fetchMe() {
      try {
        this.user = await fetchAPI('/api/auth/me');
        this.isLoggedIn = true;
        window.dispatchEvent(new CustomEvent('eco:authChanged'));
      } catch (e) {
        this.logout();
      }
    },

    /* ── Logout ───────────────────────────────────────────────────────── */
    logout() {
      this.token = null;
      this.user = null;
      this.isLoggedIn = false;
      this.loginData = { username: '', password: '' };
      this.registerData = { username: '', email: '', password: '' };
      this.passwordScore = 0;
      this.showPasswordRules = false;
      clearAuthToken();
      showToast('Te-ai deconectat cu succes.');
      window.dispatchEvent(new CustomEvent('eco:authChanged'));
    },

    openAuth(mode = 'login') {
      this.authMode = mode;
      this.showAuthModal = true;
      this.showPasswordRules = false;
    }
  };
}
