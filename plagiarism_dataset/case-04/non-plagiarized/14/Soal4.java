import java.text.DecimalFormat;
public class Soal4 {
    public static void main(String[] args) {
        int arr[][] = new int[11][2];
        DecimalFormat df = new DecimalFormat("#.###");
        for(int i = 0; i<11; i++){
            if(i==0)
                System.out.println("Miles \t Kilometers");
            else
                System.out.println(i + "\t" + df.format(i*1.609));
        }
    }
}
