public class Level6 {
    public static void konversi(int mile) {
        if (mile != 11) {
            System.out.println(mile + "\t\t" + mile * 1.609);
            mile++;
            konversi(mile);
        }
    }
    public static void main(String[] args) {
        int mile = 1;
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        konversi(mile);
    }
}
