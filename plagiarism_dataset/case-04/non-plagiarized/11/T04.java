import java.text.DecimalFormat;
public class T04 {
    public static void main(String[] args) {
        DecimalFormat d = new DecimalFormat("0.000");
        System.out.println("Miles\tKilometers");
        for(int i = 1; i <= 10; i++) {
            System.out.println(i + "\t" + d.format(i * 1.609));
        }
    }
}
