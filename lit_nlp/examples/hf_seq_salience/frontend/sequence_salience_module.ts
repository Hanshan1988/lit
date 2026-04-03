/**
 * Sequence Salience Module for HF Sequence Salience demo.
 *
 * This is a copy of client/modules/sequence_salience_module.ts adapted for
 * use in the hf_seq_salience example. To build it:
 *   1. Place this file (and the matching .css) inside client/modules/ OR
 *      symlink them from there.
 *   2. Register the element in client/main.ts.
 *   3. Run the lit_nlp TypeScript build (see the root README).
 *
 * The module renders the interactive sequence salience panel that is already
 * used by the built-in LIT UI when you open http://localhost:<port>.
 * These files are provided so you can customise or extend the visualization.
 *
 * @fileoverview LIT module for sequence salience with causal LMs.
 */

import '@material/mwc-icon';
import '../elements/color_legend';
import '../elements/interstitial';
import '../elements/numeric_input';
import '../elements/fused_button_bar';

import {css, html} from 'lit';
// tslint:disable:no-new-decorators
import {customElement, property} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';
import {computed, makeObservable, observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {NumericInput as LitNumericInput} from '../elements/numeric_input';
import {TextChips, TokenChips, TokenWithWeight} from '../elements/token_chips';
import {
  CONTINUOUS_SIGNED_LAB,
  CONTINUOUS_UNSIGNED_LAB,
  SalienceCmap,
  SignedSalienceCmap,
  UnsignedSalienceCmap,
} from '../lib/colors';
import {
  GENERATION_TYPES,
  getAllTargetOptions,
  TargetOption,
  TargetSource,
} from '../lib/generated_text_utils';
import {LitType, LitTypeTypesList, Tokens, TokenScores} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {
  cleanSpmText,
  groupTokensByRegexPrefix,
  groupTokensByRegexSeparator,
} from '../lib/token_utils';
import {
  type IndexedInput,
  type Preds,
  SCROLL_SYNC_CSS_CLASS,
  type Spec,
} from '../lib/types';
import {
  cumSumArray,
  filterToKeys,
  findSpecKeys,
  groupAlike,
  makeModifiedInput,
  sumArray,
} from '../lib/utils';

import {styles} from './sequence_salience_module.css';

export function maxAbs(vals: number[]): number {
  return Math.max(...vals.map(Math.abs));
}

enum SegmentationMode {
  TOKENS = 'Tokens',
  WORDS = 'Words',
  SENTENCES = 'Sentences',
  LINES = 'Lines',
  PARAGRAPHS = '¶',
  CUSTOM = '⚙',
}

const LEGEND_INFO_TITLE_SIGNED =
    'Salience is relative to the model\'s prediction of a token. A positive ' +
    'score (more green) for a token means that token influenced the model to ' +
    'predict the selected target, whereas a negative score (more pink) means ' +
    'the token influenced the model to not predict the selected target.';

const LEGEND_INFO_TITLE_UNSIGNED =
    'Salience is relative to the model\'s prediction of a token. A larger ' +
    'score (more purple) for a token means that token was more influential ' +
    'on the model\'s prediction of the selected target.';

const MODEL_PREDS_KEY = 'modelPreds';

export class SingleExampleSingleModelModule extends LitModule {
  static override duplicateForExampleComparison = true;
  static override duplicateForModelComparison = true;

  protected readonly predsTypes: LitTypeTypesList = [LitType];

  @observable protected currentData?: IndexedInput = undefined;
  @observable protected currentPreds?: Preds = undefined;

  constructor() {
    super();
    makeObservable(this);
  }

  protected postprocessPreds(input: IndexedInput, preds: Preds): Preds {
    return preds;
  }

  protected resetState() {
    this.currentData = undefined;
    this.currentPreds = undefined;
  }

  protected async updateToSelection() {
    this.resetState();
    const input = this.selectionService.primarySelectedInputData;
    if (input == null) return;
    this.currentData = input;

    const promise = this.apiService.getPreds(
        [input],
        this.model,
        this.appState.currentDataset,
        this.predsTypes,
        [],
        `Getting predictions from ${this.model}`,
    );
    const results = await this.loadLatest(MODEL_PREDS_KEY, promise);
    if (results == null) return;

    const preds = this.postprocessPreds(input, results[0]);
    this.currentData = input;
    this.currentPreds = preds;
  }

  override firstUpdated() {
    this.reactImmediately(
        () => [
          this.selectionService.primarySelectedInputData,
          this.model,
          this.appState.currentDataset,
        ],
        () => {
          this.updateToSelection();
        },
    );
  }
}

@customElement('sequence-salience-chips')
class SequenceSalienceChips extends TextChips {
  @property({type: Boolean}) underline = false;

  override holderClass() {
    return Object.assign({}, super.holderClass(), {'underline': this.underline});
  }

  static override get styles() {
    return [
      ...TokenChips.styles,
      css`
        .text-chips.underline .salient-token {
          --underline-height: 6px;
          background-color: transparent;
          color: black;
        }
        .text-chips.dense.underline .salient-token {
          padding-bottom: var(--underline-height);
        }
        .text-chips.underline .salient-token.selected {
          outline: 1px solid var(--token-outline-color);
          --underline-height: 5px;
        }
        .text-chips.dense.underline .salient-token span {
          border-bottom: var(--underline-height) solid var(--token-bg-color);
          border-radius: 2px;
          padding-bottom: 0;
        }
        .text-chips.dense.underline .salient-token.selected span {
          border-bottom: var(--underline-height) solid var(--lit-mage-500);
        }
        .text-chips:not(.dense).underline .salient-token {
          border-bottom: var(--underline-height) solid var(--token-bg-color);
          border-radius: 2px;
          padding-bottom: 0;
        }
        .text-chips:not(.dense).underline .salient-token.selected {
          border-bottom: var(--underline-height) solid var(--lit-mage-500);
        }
      `,
    ];
  }
}

interface SalienceResults {
  [method: string]: number[];
}

const REQUEST_PENDING: unique symbol = Symbol('REQUEST_PENDING');
const CMAP_DEFAULT_RANGE = 0.4;
const DEFAULT_CUSTOM_SEGMENTATION_REGEX = '\\n+';

/** LIT sequence salience module. */
@customElement('sequence-salience-module')
export class SequenceSalienceModule extends SingleExampleSingleModelModule {
  static override title = 'Sequence Salience';
  static override numCols = 6;
  static override duplicateAsRow = true;
  // prettier-ignore
  static override template = (
      model: string,
      selectionServiceIndex: number,
      shouldReact: number,
      ) => html`<sequence-salience-module model=${model}
          .shouldReact=${shouldReact}
          selectionServiceIndex=${selectionServiceIndex}>
        </sequence-salience-module>`;

  static override get styles() {
    return [sharedStyles, styles];
  }

  override predsTypes = GENERATION_TYPES;

  @observable private segmentationMode: SegmentationMode = SegmentationMode.WORDS;
  @observable private customSegmentationRegexString = DEFAULT_CUSTOM_SEGMENTATION_REGEX;
  @observable private selectedSalienceMethod?: string = 'grad_l2';
  @observable private cmapRange = CMAP_DEFAULT_RANGE;
  @observable private denseView = true;
  @observable private vDense = false;
  @observable private underline = false;
  @observable private showSelfSalience = false;

  @observable.ref private currentTokens: string[] = [];
  @observable.ref private salienceTargetOptions: TargetOption[] = [];
  @observable private salienceTargetOption?: number = undefined;
  @observable.ref private targetSegmentSpan?: [number, number] = undefined;

  @observable
  private salienceResultCache:
      {[targetKey: string]: SalienceResults|(typeof REQUEST_PENDING)} = {};

  @computed get salienceModelName(): string { return `_${this.model}_salience`; }
  @computed get tokenizerModelName(): string { return `_${this.model}_tokenizer`; }

  private clearResultCache() { this.salienceResultCache = {}; }
  private resetTargetSpan() { this.targetSegmentSpan = undefined; }

  override resetState() {
    super.resetState();
    this.salienceTargetOptions = [];
    this.salienceTargetOption = undefined;
    this.currentTokens = [];
    this.resetTargetSpan();
    this.clearResultCache();
  }

  @computed get modifiedData(): IndexedInput|null {
    if (this.currentData == null) return null;
    if (this.salienceTargetOption === undefined) return null;
    const targetString = this.salienceTargetOptions[this.salienceTargetOption].text;
    return makeModifiedInput(this.currentData, {'target': targetString});
  }

  @computed get customSegmentationRegex(): RegExp|undefined {
    try { return RegExp(this.customSegmentationRegexString, 'g'); }
    catch { return undefined; }
  }

  @computed get currentTokenGroups(): string[][] {
    switch (this.segmentationMode) {
      case SegmentationMode.TOKENS: return this.currentTokens.map(t => [t]);
      case SegmentationMode.WORDS:
        return groupTokensByRegexPrefix(this.currentTokens, /([▁\s]+)|(?<=\n)[^\n]/g);
      case SegmentationMode.SENTENCES:
        return groupTokensByRegexPrefix(this.currentTokens, /(\n+)|((?<=\n)[^\n])|((?<=[.?!])([▁\s]+))/g);
      case SegmentationMode.LINES:
        return groupTokensByRegexSeparator(this.currentTokens, /\n+/g);
      case SegmentationMode.PARAGRAPHS:
        return groupTokensByRegexSeparator(this.currentTokens, /\n\n+/g);
      case SegmentationMode.CUSTOM:
        if (this.customSegmentationRegex === undefined) return this.currentTokens.map(t => [t]);
        return groupTokensByRegexPrefix(this.currentTokens, this.customSegmentationRegex);
      default: throw new Error(`Unsupported mode ${this.segmentationMode}`);
    }
  }

  @computed get currentSegmentOffsets(): number[] {
    return [0, ...cumSumArray(this.currentTokenGroups.map(g => g.length))];
  }

  @computed get targetTokenSpan(): number[]|undefined {
    if (this.targetSegmentSpan === undefined) return undefined;
    const [s, e] = this.targetSegmentSpan;
    const o = this.currentSegmentOffsets;
    return [o[s], o[e]];
  }

  @computed get currentSegmentTexts(): string[] {
    const segs = this.currentTokenGroups.map(t => t.join(''));
    if (this.segmentationMode === SegmentationMode.TOKENS && !this.denseView) return segs;
    return segs.map(cleanSpmText);
  }

  @computed get salienceSpecInfo(): Spec {
    const out = this.appState.getModelSpec(this.salienceModelName).output;
    return filterToKeys(out, findSpecKeys(out, TokenScores));
  }

  @computed get activeTokenSalience(): number[]|undefined {
    if (this.targetTokenSpan === undefined) return undefined;
    const c = this.salienceResultCache[this.spanToKey(this.targetTokenSpan)];
    if (c === undefined || c === REQUEST_PENDING || this.selectedSalienceMethod === undefined) return undefined;
    return (c as SalienceResults)[this.selectedSalienceMethod];
  }

  @computed get activeSalience(): number[]|undefined {
    if (this.activeTokenSalience === undefined) return undefined;
    return groupAlike(this.activeTokenSalience, this.currentTokenGroups).map(sumArray);
  }

  @computed get cmapDomain(): number {
    if (this.activeSalience === undefined) return 1;
    return Math.max(1e-3, maxAbs(this.activeSalience));
  }

  @computed get cmapGamma(): number { return this.cmapRange * (1.0 / CMAP_DEFAULT_RANGE); }

  @computed get signedSalienceCmap() {
    return new SignedSalienceCmap(this.cmapGamma, [-this.cmapDomain, this.cmapDomain], CONTINUOUS_SIGNED_LAB, [0, this.cmapRange]);
  }

  @computed get unsignedSalienceCmap() {
    return new UnsignedSalienceCmap(this.cmapGamma, [0, this.cmapDomain], CONTINUOUS_UNSIGNED_LAB, [0, this.cmapRange]);
  }

  @computed get cmap(): SalienceCmap {
    return this.selectedSalienceMethod === 'grad_dot_input' ? this.signedSalienceCmap : this.unsignedSalienceCmap;
  }

  constructor() { super(); makeObservable(this); }

  spanToKey(span: number[]) { return `${span[0]}:${span[1]}`; }

  async updateTokens() {
    this.currentTokens = [];
    const input = this.modifiedData;
    if (input == null) return;
    const promise = this.apiService.getPreds([input], this.tokenizerModelName, this.appState.currentDataset, [Tokens], [], `Fetching tokens for ${this.model}`);
    const results = await this.loadLatest('updateTokens', promise);
    if (results === null) return;
    this.currentTokens = results[0]['tokens'];
  }

  async fetchSalience(span: number[]|undefined) {
    if (this.modifiedData == null || span === undefined) return;
    const key = this.spanToKey(span);
    const cached = this.salienceResultCache[key];
    if (cached !== undefined) return;
    this.salienceResultCache[key] = REQUEST_PENDING;
    const [start, end] = span;
    const mask = this.currentTokens.map((_, i) => (i >= start && i < end) ? 1 : 0);
    const maskedData = makeModifiedInput(this.modifiedData, {'target_mask': mask}, 'salience');
    const promise = this.apiService.getPreds([maskedData], this.salienceModelName, this.appState.currentDataset, [TokenScores], [], `Getting salience for ${this.printTargetForHuman(start, end)}`);
    const results = await promise;
    if (results === null) { delete this.salienceResultCache[key]; return; }
    this.salienceResultCache[key] = results[0];
  }

  override firstUpdated() {
    super.firstUpdated();
    this.reactImmediately(
        () => [this.currentData, this.currentPreds] as const,
        ([data, preds]) => {
          this.salienceTargetOptions = getAllTargetOptions(this.appState.currentDatasetSpec, this.appState.getModelSpec(this.model).output, data, preds);
        });
    this.reactImmediately(
        () => [this.modifiedData, this.model, this.appState.currentDataset],
        () => { this.clearResultCache(); this.resetTargetSpan(); this.updateTokens(); });
    this.reactImmediately(
        () => this.targetTokenSpan,
        (span) => { this.fetchSalience(span); });
  }

  /* ── render helpers ── */

  renderLoadingIndicator() {
    return html`<div class='loading-indicator-container'><div class='loading-indicator'></div></div>`;
  }

  renderGranularitySelector() {
    const segOpts = Object.values(SegmentationMode).map(v => ({
      text: v,
      selected: this.segmentationMode === v,
      tooltipText: v === SegmentationMode.PARAGRAPHS ? 'Paragraphs' : v === SegmentationMode.CUSTOM ? 'Custom Regex' : '',
      onClick: () => { if (this.segmentationMode !== v) this.resetTargetSpan(); this.segmentationMode = v as SegmentationMode; },
    }));
    const regexClasses = classMap({'regex-input': true, 'error-input': this.customSegmentationRegex === undefined});
    const customEntry = html`<div class='regex-input-container'><input type='text' class=${regexClasses} .value=${this.customSegmentationRegexString} @input=${(e: Event) => { this.customSegmentationRegexString = (e.target as HTMLInputElement).value; this.resetTargetSpan(); }}><mwc-icon class='icon-button' @click=${() => { this.customSegmentationRegexString = DEFAULT_CUSTOM_SEGMENTATION_REGEX; }}>restart_alt</mwc-icon></div>`;
    return html`
      <div class="controls-group" style="gap: 8px;">
        <label class="dropdown-label" id="granularity-label">Granularity:</label>
        <lit-fused-button-bar .options=${segOpts} ?disabled=${this.currentTokens.length === 0}></lit-fused-button-bar>
        ${this.segmentationMode === SegmentationMode.CUSTOM ? customEntry : null}
      </div>
      <div class="controls-group" style="gap: 8px;">
        <lit-switch ?selected=${!this.denseView} @change=${() => { this.denseView = !this.denseView; }}>
          <mwc-icon class='icon-button large-icon' slot='labelLeft' title='Flowing text'>notes</mwc-icon>
          <mwc-icon class='icon-button large-icon' slot='labelRight' title='Segments'>grid_view</mwc-icon>
        </lit-switch>
        <mwc-icon class='icon-button large-icon' @click=${() => { this.vDense = !this.vDense; }}>
          ${this.vDense ? 'expand' : 'compress'}
        </mwc-icon>
        <mwc-icon class='icon-button large-icon' @click=${() => { this.underline = !this.underline; }}>
          ${this.underline ? 'font_download' : 'format_color_text'}
        </mwc-icon>
      </div>
      <div class='flex-grow-spacer'></div>`;
  }

  renderSelfScoreSelector() { return null; }

  renderMethodSelector() {
    const opts = Object.keys(this.salienceSpecInfo).map(key => ({
      text: key,
      selected: this.selectedSalienceMethod === key,
      onClick: () => { if (this.selectedSalienceMethod !== key) this.selectedSalienceMethod = key; },
    }));
    return html`<div class="controls-group" style="gap: 8px;"><label class="dropdown-label" id="method-label">Method:</label><lit-fused-button-bar .options=${opts}></lit-fused-button-bar>${this.renderSelfScoreSelector()}</div>`;
  }

  targetSpanText(start: number, end: number): string {
    const tokens = this.currentTokens.slice(start, end);
    if (this.segmentationMode === SegmentationMode.TOKENS && !this.denseView) return tokens.join(' ');
    return cleanSpmText(tokens.join('')).trim();
  }

  printTargetForHuman(start: number, end: number): string {
    const text = this.targetSpanText(start, end);
    return end === start + 1 ? `[${start}] "${text}"` : `[${start}:${end}] "${text}"`;
  }

  renderSalienceTargetStringIndicator() {
    const target = this.salienceTargetOption !== undefined ? this.salienceTargetOptions[this.salienceTargetOption] : null;
    const sourceInfo = target ? (target.source === TargetSource.REFERENCE ? ' (target)' : ' (response)') : '';
    const text = target ? target.text : 'none selected.';
    const loading = this.latestLoadPromises.has(MODEL_PREDS_KEY);
    const cls = classMap({'target-info-line': true, 'gray-text': target == null});
    return html`
      <div class="controls-group controls-group-variable">
        <label class="dropdown-label">Sequence${sourceInfo}:</label>
        <div class=${cls} title=${text}>${text}${loading ? this.renderLoadingIndicator() : null}</div>
      </div>
      <div class='controls-group'>
        <button class='hairline-button' @click=${() => { this.salienceTargetOption = undefined; }} ?disabled=${target == null}>
          <span>Select sequence </span><span class='material-icon'>arrow_drop_down</span>
        </button>
      </div>`;
  }

  renderTargetIndicator() {
    const printTarget = () => {
      if (this.targetTokenSpan === undefined) {
        const type = this.segmentationMode === SegmentationMode.TOKENS ? 'token(s)' : 'segment(s)';
        return html`<span class="gray-text">Click ${type} above to select a target to explain.</span>`;
      }
      const [s, e] = this.targetTokenSpan;
      return `Explaining ${this.printTargetForHuman(s, e)}`;
    };
    const pending = this.targetTokenSpan !== undefined && this.salienceResultCache[this.spanToKey(this.targetTokenSpan)] === REQUEST_PENDING;
    const cls = classMap({'target-info-line': true, 'gray-text': pending});
    return html`<div class="controls-group controls-group-variable"><div class=${cls}>${printTarget()}${pending ? this.renderLoadingIndicator() : null}</div></div>`;
  }

  private setSegmentTarget(i: number, shift = false) {
    if (this.targetSegmentSpan === undefined) {
      this.targetSegmentSpan = [i, i + 1]; return;
    }
    const [s, e] = this.targetSegmentSpan;
    if (shift) {
      if (i < s) this.targetSegmentSpan = [i, e];
      else if (i >= e) this.targetSegmentSpan = [s, i + 1];
    } else {
      if (i === s - 1) this.targetSegmentSpan = [i, e];
      else if (i === e) this.targetSegmentSpan = [s, i + 1];
      else if (i === s) this.targetSegmentSpan = s + 1 < e ? [s + 1, e] : undefined;
      else if (i === e - 1) this.targetSegmentSpan = s < e - 1 ? [s, e - 1] : undefined;
      else this.targetSegmentSpan = [i, i + 1];
    }
  }

  private inTargetSpan(i: number) {
    if (this.targetSegmentSpan === undefined) return false;
    return i >= this.targetSegmentSpan[0] && i < this.targetSegmentSpan[1];
  }

  renderTargetSelectorInterstitial() {
    const fmtOpt = (t: TargetOption, i: number) => ({
      source: t.source,
      tpl: html`<div class='interstitial-target-option' @click=${() => { this.salienceTargetOption = i; }}><div class='interstitial-target-text'>${t.text}</div></div>`,
    });
    const opts = this.salienceTargetOptions.map((t, i) => fmtOpt(t, i));
    const loading = this.latestLoadPromises.has(MODEL_PREDS_KEY);
    return html`
      <div class='interstitial-container'>
        <div class='interstitial-contents'>
          <div class='interstitial-header'>Choose a sequence to explain</div>
          <div class='interstitial-target-selector'>
            <div class='interstitial-target-type'>From dataset (target):</div>
            ${opts.filter(o => o.source === TargetSource.REFERENCE).map(o => o.tpl)}
            <div class='interstitial-target-type'>From model (response):</div>
            ${loading ? this.renderLoadingIndicator() : null}
            ${opts.filter(o => o.source === TargetSource.MODEL_OUTPUT).map(o => o.tpl)}
          </div>
        </div>
      </div>`;
  }

  renderNoExampleInterstitial() {
    return html`<lit-interstitial headline="Sequence Salience">Enter a prompt in the Editor or select an example from the Data Table to begin.</lit-interstitial>`;
  }

  renderContent() {
    if (this.currentData == null) return this.renderNoExampleInterstitial();
    if (this.salienceTargetOption === undefined) return this.renderTargetSelectorInterstitial();
    if (this.currentSegmentTexts.length === 0) return null;

    const segs = this.currentSegmentTexts;
    const items: TokenWithWeight[] = segs.map((tok, i) => {
      const selected = this.inTargetSpan(i);
      let w = this.activeSalience?.[i] ?? 0;
      if (selected && !this.showSelfSalience) w = 0;
      return {
        token: tok, weight: w, selected,
        onClick: (e: MouseEvent) => {
          this.setSegmentTarget(i, e.shiftKey);
          if (e.shiftKey) document.getSelection()?.removeAllRanges();
          e.stopPropagation();
        },
      };
    });

    return html`<div class='chip-container'><sequence-salience-chips .tokensWithWeights=${items} .cmap=${this.cmap} ?dense=${this.denseView} ?vDense=${this.vDense} ?underline=${this.underline} ?preSpace=${this.denseView} breakNewlines></sequence-salience-chips></div>`;
  }

  renderColorLegend() {
    const cmap = this.cmap;
    const signed = cmap instanceof SignedSalienceCmap;
    return html`<color-legend legendType=${LegendType.SEQUENTIAL} label="Salience" .scale=${cmap.asScale()} numBlocks=${signed ? 7 : 5} tooltipPosition="above" paletteTooltipText=${signed ? LEGEND_INFO_TITLE_SIGNED : LEGEND_INFO_TITLE_UNSIGNED}></color-legend>`;
  }

  renderColorControls() {
    const onChange = (e: Event) => { this.cmapRange = Number((e.target as LitNumericInput).value); };
    return html`<div class="controls-group">${this.renderColorLegend()}<mwc-icon class='icon'>opacity</mwc-icon><label id='colormap-slider-label' class='dropdown-label'>Colormap intensity:</label><lit-numeric-input min="0" max="1" step="0.1" value="${this.cmapRange}" @change=${onChange}></lit-numeric-input><mwc-icon class='icon-button' @click=${() => { this.cmapRange = CMAP_DEFAULT_RANGE; }}>restart_alt</mwc-icon></div>`;
  }

  override renderImpl() {
    return html`
      <div class="module-container">
        <div class="module-toolbar">${this.renderSalienceTargetStringIndicator()}</div>
        <div class="module-toolbar">${this.renderGranularitySelector()}${this.renderMethodSelector()}</div>
        <div class="module-results-area ${SCROLL_SYNC_CSS_CLASS} flex-column" @click=${() => { this.resetTargetSpan(); }}>
          ${this.renderContent()}
        </div>
        <div class="module-footer module-footer-wrappable">
          ${this.renderTargetIndicator()}${this.renderColorControls()}
        </div>
      </div>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'sequence-salience-module': SequenceSalienceModule;
  }
}
