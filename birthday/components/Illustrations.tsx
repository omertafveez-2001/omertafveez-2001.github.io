export function Cake() {
  return <svg className="cake" viewBox="0 0 360 290" fill="none" role="img" aria-label="A little pink birthday cake with a glowing candle">
    <ellipse cx="180" cy="259" rx="120" ry="12" fill="#8B4557" opacity=".07"/>
    <path d="M60 238Q180 268 300 238L291 252Q180 280 69 252Z" fill="#E7CCD0"/>
    <ellipse cx="180" cy="236" rx="120" ry="22" fill="#FFFCF8" stroke="#C799A5" strokeWidth="1.5"/>
    <path d="M83 148V225C83 251 277 251 277 225V148" fill="#EBA4B8" stroke="#B77589" strokeWidth="1.5"/>
    <path d="M88 200Q180 224 272 200M88 210Q180 234 272 210" stroke="#F9CBDA" strokeWidth="4"/>
    <ellipse cx="180" cy="147" rx="97" ry="29" fill="#FFF4ED" stroke="#B77589" strokeWidth="1.5"/>
    <path d="M83 148C84 164 96 162 100 175C105 192 115 191 119 172C122 160 137 167 141 180C147 201 158 195 159 179C160 167 179 176 186 175C195 174 202 165 207 179C213 196 224 186 225 174C227 160 241 161 245 169C253 184 263 176 264 163C265 157 277 158 277 148" fill="#FFF4ED" stroke="#B77589" strokeWidth="1.5"/>
    <ellipse cx="180" cy="142" rx="77" ry="18" stroke="#E7C6CC" strokeDasharray="2 5"/>
    <path d="M173 132V76Q180 72 187 76V132Q180 136 173 132Z" fill="#F7D2DB" stroke="#B77589" strokeWidth="1.5"/>
    <path d="M174 89L186 82M174 105L186 98M174 121L186 114" stroke="#D591AA" strokeWidth="3"/>
    <path d="M180 74V65" stroke="#8B4557" strokeWidth="2"/>
    <g className="candle-flame"><ellipse cx="180" cy="49" rx="25" ry="30" fill="#F6CF8E" opacity=".2"/><path d="M180 26C181 38 195 46 187 59C182 68 169 63 170 53C170 42 177 37 180 26Z" fill="#E8B573"/><path d="M180 43C180 50 185 52 182 58C177 63 174 55 180 43Z" fill="#FFF6C9"/></g>
    <g stroke="#B97F96" strokeWidth="1.4" strokeLinecap="round"><path d="M54 95V109M47 102H61M300 139V153M293 146H307M265 59V69M260 64H270"/></g>
  </svg>;
}

export function Vanity({ laughs }: { laughs: number }) {
  return <svg className="vanity" viewBox="0 0 420 300" fill="none" role="img" aria-label="An illustrated skincare vanity with a pink mirror, bottles, and a tiny smiling face">
    <ellipse cx="210" cy="273" rx="171" ry="13" fill="#8B4557" opacity=".06"/>
    <path d="M60 252H360L349 265H71Z" fill="#CE9AA8"/><rect x="52" y="240" width="316" height="14" rx="7" fill="#F0CCD2" stroke="#B67C8D"/>
    <path d="M123 228V116C123 5 297 5 297 116V228Z" fill="#EDC3D2" stroke="#B67C8D" strokeWidth="2"/>
    <path d="M135 220V116C135 20 285 20 285 116V220Z" fill="#F9EEE9" stroke="#B67C8D"/>
    <path d="M147 175L251 61M159 195L274 69" stroke="white" strokeWidth="9" opacity=".65"/>
    <ellipse cx="210" cy="157" rx="38" ry="43" fill="#F2D5C4"/>
    <path d="M172 155C161 94 245 88 249 150C234 142 234 118 225 119C214 143 185 139 172 155Z" fill="#79584E"/>
    <path d="M192 156Q197 151 202 156M220 156Q225 151 230 156" stroke="#79584E" strokeWidth="2" strokeLinecap="round"/>
    <path d={laughs ? 'M200 173Q211 190 223 173Z' : 'M204 175Q212 182 220 175'} fill={laughs ? '#B86B7C' : 'none'} stroke="#B86B7C" strokeWidth="2" strokeLinecap="round"/>
    <ellipse cx="189" cy="170" rx="7" ry="4" fill="#E9A9AF"/><ellipse cx="235" cy="170" rx="7" ry="4" fill="#E9A9AF"/>
    <path d="M174 220Q177 194 191 195Q210 210 229 195Q244 194 247 220" fill="#D7C7E5"/>
    <rect x="76" y="175" width="37" height="65" rx="7" fill="#DCD1E8" stroke="#A28AB7"/><rect x="82" y="160" width="25" height="18" rx="3" fill="#A28AB7"/><path d="M92 160V151H112" stroke="#A28AB7" strokeWidth="4" strokeLinecap="round"/><path d="M87 202H103M90 208H100" stroke="white" strokeWidth="2"/>
    <rect x="305" y="192" width="39" height="48" rx="5" fill="#F7E3CB" stroke="#C29C79"/><path d="M309 192V180H340V192" fill="#E6C8AB" stroke="#C29C79"/><path d="M316 209H334M319 216H331" stroke="#C29C79" strokeWidth="1.5"/>
    <rect x="264" y="216" width="31" height="24" rx="5" fill="#F3C9D7" stroke="#B67C8D"/><path d="M264 220H295" stroke="#B67C8D"/>
    <path d="M58 120V132M52 126H64M329 108V120M323 114H335" stroke="#C69AAE" strokeWidth="1.5" strokeLinecap="round"/>
  </svg>;
}
