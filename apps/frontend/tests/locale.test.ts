import { describe, expect, it } from 'vitest';

import { detectPreferredLocale, getHtmlLang, resolveLocale } from '@/lib/i18n/locale';

describe('locale helpers', () => {
  it('resolves region-specific language tags', () => {
    expect(resolveLocale('zh-CN')).toBe('zh');
    expect(resolveLocale('pt-BR')).toBe('pt');
    expect(resolveLocale('en_US')).toBe('en');
  });

  it('detects the first supported browser locale', () => {
    expect(detectPreferredLocale(['fr-FR', 'zh-CN', 'en-US'])).toBe('zh');
    expect(detectPreferredLocale(['de-DE', 'it-IT'])).toBe('en');
  });

  it('maps locales to html lang tags', () => {
    expect(getHtmlLang('zh')).toBe('zh-CN');
    expect(getHtmlLang('pt')).toBe('pt-BR');
  });
});
